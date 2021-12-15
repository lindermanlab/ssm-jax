import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
import flax.linen as nn
from copy import deepcopy as dc
from jax import flatten_util

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals

from ssm.lds.models import GaussianLDS

from colormap import hex2rgb
muted_colours_list = ["#4878D0", "#D65F5F", "#EE854A", "#6ACC64", "#956CB4",
                      "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
muted_colours_list = np.asarray([hex2rgb(_c) for _c in muted_colours_list]) / 256
muted_colours_dict = {'blue':   muted_colours_list[0],
                      'red':    muted_colours_list[1],
                      'orange': muted_colours_list[2],
                      'green':  muted_colours_list[3],
                      'purple': muted_colours_list[4],
                      'brown':  muted_colours_list[5],
                      'pink':   muted_colours_list[6],
                      'gray':   muted_colours_list[7],
                      'yellow': muted_colours_list[8],
                      'eggsh':  muted_colours_list[9]}
mcd = muted_colours_dict
cols = muted_colours_list


def lds_define_test(subkey):

    # Define the true model.
    key, subkey = jr.split(subkey)
    true_model, true_states, dataset = lds_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = lds_define_test_model(subkey, true_model)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = lds_define_proposal(subkey, model, dataset)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, dataset)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    return ret_model, ret_test, ret_prop


def lds_define_test_model(subkey, true_model, ):
    """

    :param subkey:
    :param true_model:
    :return:
    """

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    free_parameters = ('dynamics_weights', )

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, free_parameters)

    if len(free_parameters) > 0:

        # Get the default parameters from the true model.
        true_params = fivo.get_model_params_fn(true_model)

        # Generate a model to use.  NOTE - this will generate a new model, and we will
        # overwrite any of the free parameters of interest into the true model.
        tmp_model = true_model.__class__(num_latent_dims=true_model.latent_dim,
                                         num_emission_dims=true_model.emissions_shape[0],
                                         seed=subkey)

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def lds_define_proposal(subkey, model, dataset):
    """

    :param subkey:
    :param model:
    :param dataset:
    :return:
    """

    proposal_structure = 'RESQ'         # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    proposal_type = 'SHARED'            # {'SHARED', 'INDePENDENT'}

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = flatten_util.ravel_pytree(np.ones((model.latent_dim,)), )[0]
    output_dim = flatten_util.ravel_pytree(dummy_proposal_output)[0].shape[0]

    # # Define some different link functions.
    # trunk_fn = None  # MLP(features=(3, 4, 5), kernel_init=w_init)

    w_init_mean = lambda *args: (0.1 * jax.nn.initializers.normal()(*args))
    head_mean_fn = nn.Dense(output_dim, kernel_init=w_init_mean)

    w_init_mean = lambda *args: ((0.1 * jax.nn.initializers.normal()(*args)) - 3)
    # head_log_var_fn = nn.Dense(output_dim, kernel_init=w_init_mean)
    head_log_var_fn = nn_util.Static(output_dim, kernel_init=w_init_mean)

    # Define the number of proposals to define depending on the proposal type.
    if proposal_type == 'SHARED':
        n_proposals = 1
    else:
        n_proposals = len(dataset)

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_proposals,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def lds_define_true_model_and_data(key):
    """

    :param key:
    :return:
    """

    # Define the model parameters.
    latent_dim = 3
    emissions_dim = 5
    num_trials = 5
    num_timesteps = 100

    # Create a more reasonable emission scale.
    transition_scale_tril = 0.1 * np.eye(latent_dim)
    emission_scale_tril = 0.5 * np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)
    true_dynamics_weights = random_rotation(subkey, latent_dim, theta=np.pi / 10)
    key, subkey = jr.split(key)
    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=transition_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    return true_model, true_states, dataset


def lds_do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params, param_figs):

    fsize = (12, 8)
    idx_to_str = lambda _idx: ['Model (p): ', 'Proposal (q): '][_idx]

    # Plot the loss.
    if param_figs[0] is None:
        param_figs[0] = plt.figure(figsize=fsize)
    else:
        plt.figure(param_figs[0].number)
    plt.cla()
    plt.plot(_loss_hist, label='Cur LML (FIVO)')
    plt.plot([0, len(_loss_hist) - 1], [_true_loss_em, _true_loss_em], '--', c='k', label='True lml (EM)')
    plt.plot([0, len(_loss_hist) - 1], [_true_loss_smc, _true_loss_smc], ':', c='k', linewidth=2.0, label='True lml (SMC)')
    plt.legend()
    plt.title('FIVO Loss.')
    plt.title('Step')
    plt.tight_layout()
    plt.pause(0.0001)

    # Plot the parameters.
    for _p, _hist in enumerate(_param_hist):

        if len(_hist[0]) > 0:

            n_param = len(_hist[0].keys())

            if param_figs[_p + 1] is None:
                param_figs[_p + 1] = plt.subplots(n_param, 1, figsize=fsize, sharex=True, squeeze=True)

            for _i, _k in enumerate(_hist[0].keys()):
                to_plot = []
                for _pt in _param_hist[_p]:
                    to_plot.append(_pt[_k].flatten())
                to_plot = np.array(to_plot)

                if hasattr(param_figs[_p + 1][1], '__len__'):
                    plt.sca(param_figs[_p + 1][1][_i])
                else:
                    plt.sca(param_figs[_p + 1][1])
                plt.cla()
                plt.pause(0.00001)

                # Plot the predicted parameters.
                for __idx, __pv in enumerate(to_plot.T):
                    if __idx == 0:
                        plt.plot(__pv, label='Current parameters', c=cols[__idx], )
                    else:
                        plt.plot(__pv, c=cols[__idx], )
                if _p == 0:
                    # Plot the true model parameters.
                    for __idx, __pv in enumerate(_true_params[_p].flatten()):
                        if __idx == 0:
                            plt.plot([0, len(to_plot)-1], [__pv, __pv], '--', c=cols[__idx], label='True parameters')
                        else:
                            plt.plot([0, len(to_plot)-1], [__pv, __pv], '--', c=cols[__idx], )

                plt.title(idx_to_str(_p) + str(_k))
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.pause(0.00001)

    return param_figs


def lds_do_print(step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
    """
    Do menial print stuff.
    :param _step:
    :param pred_lml:
    :param true_model:
    :param true_lml:
    :param opt:
    :param em_log_marginal_likelihood:
    :return:
    """

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},'.format(step, true_lml, )
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)
    _str += '  Pred Neg-LML: {: >8.3f}'.format(pred_lml)
    _str += '  Pred FIVO-bound: {: >8.3f}'.format(pred_fivo_bound)
    print(_str)

    if opt[0] is not None:
        print('True: dynamics:  ', '  '.join(['{: >9.3f}'.format(_s) for _s in true_model.dynamics_matrix.flatten()]))
        print('Pred: dynamics:  ', '  '.join(['{: >9.3f}'.format(_s) for _s in opt[0].target[0].flatten()]))

    # if opt[1] is not None:
    #     print('True: log-var:   ', '  '.join(['{: >9.5f}'.format(_s) for _s in np.log(np.diagonal(true_model.dynamics_noise_covariance))]))
    #     print('Pred: q log-var: ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[1].target._asdict()['head_log_var_fn'].W.flatten()]))
    print()
