import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import argparse
from jax import random as jr
import flax.linen as nn
from typing import NamedTuple
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def gdm_get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GDM', type=str)

    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--log-group', default='debug', type=str)               # {'debug', 'gdm-v1.0'}

    parser.add_argument('--use-sgr', default=1, type=int)                       # {0, 1}

    parser.add_argument('--temper', default=0.0, type=float)  # {0.0 to disable,  >0.1 to temper}.

    parser.add_argument('--free-parameters', default='dynamics_bias', type=str)              # CSV.  # 'dynamics_bias'

    parser.add_argument('--proposal-structure', default='DIRECT', type=str)       # {None/'BOOTSTRAP', 'DIRECT', 'RESQ', }
    parser.add_argument('--proposal-type', default='PERSTEP', type=str)  # {'PERSTEP', }.

    parser.add_argument('--tilt-structure', default='DIRECT', type=str)         # {None/'NONE', 'DIRECT'}
    parser.add_argument('--tilt-type', default='SINGLEWINDOW', type=str)  # {'SINGLEWINDOW', 'PERSTEPWINDOW', 'PERSTEP'}.

    parser.add_argument('--num-particles', default=5, type=int)
    parser.add_argument('--datasets-per-batch', default=64, type=int)
    parser.add_argument('--opt-steps', default=100000, type=int)

    parser.add_argument('--p-lr', default=0.001, type=float)
    parser.add_argument('--q-lr', default=0.001, type=float)
    parser.add_argument('--r-lr', default=0.001, type=float)

    parser.add_argument('--T', default=9, type=int)   # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.
    parser.add_argument('--latent-dim', default=1, type=int)
    parser.add_argument('--emissions-dim', default=1, type=int)
    parser.add_argument('--num-trials', default=100000, type=int)

    parser.add_argument('--dset-to-plot', default=2, type=int)
    parser.add_argument('--num-val-datasets', default=1000, type=int)
    parser.add_argument('--validation-particles', default=250, type=int)
    parser.add_argument('--sweep-test-particles', default=10, type=int)

    parser.add_argument('--load-path', default=None, type=str)  # './params_lds_tmp.p'
    parser.add_argument('--save-path', default=None, type=str)  # './params_lds_tmp.p'

    parser.add_argument('--PLOT', default=1, type=int)

    config = parser.parse_args().__dict__

    # Make sure this one is formatted correctly.
    config['model'] = 'GDM'

    # Do some checking.
    assert config['latent-dim'] == 1
    assert config['emissions-dim'] == 1

    return config


def gdm_define_test(key, free_parameters, proposal_structure, tilt_structure):

    # Define the true model.
    key, subkey = jr.split(key)
    true_model, true_states, dataset = gdm_define_true_model_and_data(subkey)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = gdm_define_test_model(subkey, true_model, free_parameters)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = gdm_define_proposal(subkey, model, dataset, proposal_structure)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = gdm_define_tilt(subkey, model, dataset, tilt_structure)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, dataset)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    return ret_model, ret_test, ret_prop, ret_tilt


def gdm_define_test_model(key, true_model, free_parameters):
    """

    :param subkey:
    :param true_model:
    :return:
    """
    key, subkey = jr.split(key)

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

        # Mutate the free parameters.
        for _k in free_parameters:
            _base = getattr(default_params, _k)
            key, subkey = jr.split(key)
            new_val = {_k: _base + (10.0 * jr.normal(key=subkey, shape=_base.shape))}
            default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


class GdmTilt(tilts.IndependentGaussianTilt):

    def apply(self, params, dataset, model, particles, t, *inputs):
        """

        Args:
            params (FrozenDict):    FrozenDict of the parameters of the tilt.

            dataset:

            model:

            particles:

            t:

            inputs (tuple):         Tuple of additional inputs to the tilt in SMC.

            data:

        Returns:
            (Float): Tilt log value.

        """

        # Pull out the time and the appropriate tilt.
        if self.n_tilts == 1:
            t_params = params[0]
        else:
            t_params = jax.tree_map(lambda args: args[t], params)

        # Generate a tilt distribution.
        tilt_inputs = self._tilt_input_generator(dataset, model, particles, t, *inputs)
        r_dist = self.tilt.apply(t_params, tilt_inputs)

        # # Force optimal tilt here for default GDM example.
        # r_dist = tfd.MultivariateNormalDiag(loc=tilt_inputs, scale_diag=np.sqrt(r_dist.variance()))

        # Now score under that distribution.
        tilt_outputs = self._tilt_output_generator(dataset, model, particles, t, *inputs)
        log_r_val = r_dist.log_prob(tilt_outputs)

        return log_r_val

    # We need to define the method for generating the inputs.
    @staticmethod
    def _tilt_output_generator(dataset, model, particles, t, *_inputs):
        """
        Define the output generator for the gdm example.
        Args:
            dataset:

            model:

            particles:

            t:

            *inputs_:

        Returns:

        """

        tilt_inputs = (dataset[-1], )  # Just the data are passed in.
        return nn_util.vectorize_pytree(tilt_inputs)


def gdm_define_tilt(subkey, model, dataset, tilt_structure):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if (tilt_structure is None) or (tilt_structure == 'NONE'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Check whether we have a valid number of tilts.
    n_tilts = len(dataset[0]) - 1

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0)
    dummy_tilt_output = nn_util.vectorize_pytree(dataset[0][-1], )

    # Define a more conservative initialization.
    w_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    b_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    head_mean_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=w_init, bias_init=b_init)

    # b_init = lambda *args: ((0.1 * jax.nn.initializers.normal()(*args) + 1))  # For when not using variance
    b_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    head_log_var_fn = nn_util.Static(dummy_tilt_output.shape[0], bias_init=b_init)

    # Define the tilt itself.
    tilt = GdmTilt(n_tilts=n_tilts,
                   tilt_input=stock_tilt_input,
                   head_mean_fn=head_mean_fn,
                   head_log_var_fn=head_log_var_fn)

    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, tilt_structure)
    return tilt, tilt_params, rebuild_tilt_fn


def gdm_define_proposal(subkey, model, dataset, proposal_structure):
    """

    :param subkey:
    :param model:
    :param dataset:
    :return:
    """

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    # Define a more conservative initialization.
    w_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    b_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    head_mean_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=w_init, bias_init=b_init)

    # b_init = lambda *args: ((0.1 * jax.nn.initializers.normal()(*args) + 1))  # For when not using variance
    b_init = lambda *args: (10.0 * jax.nn.initializers.normal()(*args))
    # head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=w_init, bias_init=b_init)
    head_log_var_fn = nn_util.Static(dummy_proposal_output.shape[0], bias_init=b_init)

    # Check whether we have a valid number of proposals.
    n_props = len(dataset[0])

    # Define the required method for building the inputs.
    def gdm_proposal_input_generator(*_inputs):
        """
        Converts inputs of the form (dataset, model, particle[SINGLE], t, p_dist, q_state) into a vector object that
        can be input into the proposal.

        Args:
            *_inputs (tuple):       Tuple of standard inputs to the proposal in SMC:
                                    (dataset, model, particles, time, p_dist)

        Returns:
            (ndarray):              Processed and vectorized version of `*_inputs` ready to go into proposal.

        """

        dataset, _, particles, t, _, _ = _inputs  # NOTE - this part of q can't actually use model or p_dist.

        proposal_inputs = (jax.lax.dynamic_index_in_dim(_inputs[0], index=len(dataset)-1, axis=0, keepdims=False),
                           _inputs[2])

        is_batched = (_inputs[1].latent_dim != particles.shape[0])
        if not is_batched:
            return nn_util.vectorize_pytree(proposal_inputs)
        else:
            vmapped = jax.vmap(nn_util.vectorize_pytree, in_axes=(None, 0))
            return vmapped(*proposal_inputs)

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_props,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     input_generator=gdm_proposal_input_generator,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )

    # Initialize the network.
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def gdm_get_true_target_marginal(model, data):
    """
    Take in a model and some data and return the tfd distribution representing the marginals of true posterior.
    Args:
        model:
        data:

    Returns:

    """

    #
    # NOTE - this assumes that `\alpha = 0`.
    #

    assert len(data.shape) == 3

    T = data.shape[1] - 1
    t = np.arange(0, T + 1)
    sigma_p_sq = model.initial_covariance.squeeze()
    sigma_f_sq = model.dynamics_noise_covariance.squeeze()
    sigma_x_sq = model.emissions_noise_covariance.squeeze()
    mu_p = model.initial_mean.squeeze()
    obs = data[:, -1, :].squeeze()

    precision_1 = 1.0 / (sigma_p_sq + t * sigma_f_sq)
    precision_2 = 1.0 / (sigma_x_sq + (T - t) * sigma_f_sq)

    sigma_sq = 1.0 / (precision_1 + precision_2)
    mu = (((mu_p * precision_1) + (np.expand_dims(obs, 1) * precision_2)) * sigma_sq)

    dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=np.sqrt(sigma_sq))

    assert dist.batch_shape == data.shape[0]
    assert dist.event_shape == data.shape[1]

    return dist


def gdm_define_true_model_and_data(key):
    """

    :param key:
    :return:
    """
    latent_dim = 1
    emissions_dim = 1
    num_trials = 100000
    T = 9  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.

    # Create a more reasonable emission scale.
    dynamics_scale_tril = 1.0 * np.eye(latent_dim)
    true_dynamics_weights = np.eye(latent_dim)
    true_emission_weights = np.eye(emissions_dim)

    # NOTE - can make observations tighter here.
    emission_scale_tril = 0.1 * np.eye(emissions_dim)
    # emission_scale_tril = 1.0 * np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)

    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=dynamics_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_weights=true_emission_weights,
                             emission_scale_tril=emission_scale_tril,
                             )

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=T+1, num_samples=num_trials)

    # For the GDM example we zero out all but the last elements.
    dataset = dataset.at[:, :-1].set(np.nan)

    return true_model, true_states, dataset


def gdm_do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params,
                param_figs):

    fsize = (12, 8)
    idx_to_str = lambda _idx: ['Model (p): ', 'Proposal (q): ', 'Tilt (r): '][_idx]

    for _p, _hist in enumerate(_param_hist[:3]):

        if _hist[0] is not None:
            if len(_hist[0]) > 0:

                n_param = len(_hist[0].keys())

                if param_figs[_p] is None:
                    param_figs[_p] = plt.subplots(n_param, 1, figsize=fsize, sharex=True, squeeze=True)

                for _i, _k in enumerate(_hist[0].keys()):
                    to_plot = []
                    for _pt in _param_hist[_p]:
                        to_plot.append(_pt[_k].flatten())
                    to_plot = np.array(to_plot)

                    if hasattr(param_figs[_p][1], '__len__'):
                        plt.sca(param_figs[_p][1][_i])
                    else:
                        plt.sca(param_figs[_p][1])
                    plt.cla()
                    plt.plot(to_plot)
                    plt.title(idx_to_str(_p) + str(_k))
                    plt.grid(True)
                    plt.tight_layout()
                    plt.pause(0.00001)

                plt.savefig('./gdm_param_{}.pdf'.format(_p))

    return param_figs


def gdm_do_print(_step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
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

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f},  Pred FIVO bound {: >8.3f}'.\
        format(_step, true_lml, pred_lml, pred_fivo_bound)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    if opt[0] is not None:
        if len(opt[0].target) > 0:
            # print()
            print('\tModel')
            true_bias = true_model.dynamics_bias.flatten()
            pred_bias = opt[0].target[0].flatten()
            print('\t\tTrue: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in true_bias]))
            print('\t\tPred: dynamics bias:     ', '  '.join(['{: >9.3f}'.format(_s) for _s in pred_bias]))

    if opt[2] is not None:
        r_param = opt[2].target._dict['params']
        print('\tTilt:')

        r_mean_w = r_param['head_mean_fn']['kernel']
        print('\t\tR mean weight:           ', '  '.join(['{: >9.3f}'.format(_s) for _s in r_mean_w.flatten()]))

        try:
            r_mean_b = r_param['head_mean_fn']['bias']  # ADD THIS BACK IF WE ARE USING THE BIAS
            print('\t\tR mean bias       (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in (np.exp(r_mean_b.flatten()))]))
        except:
            pass

        try:
            r_lvar_w = r_param['head_log_var_fn']['kernel']
            print('\t\tR var(log) kernel (->0):', '  '.join(['{: >9.3f}'.format(_s) for _s in r_lvar_w.flatten()]))
        except:
            pass

        r_lvar_b = r_param['head_log_var_fn']['bias']
        print('\t\tR var bias:              ', '  '.join(['{: >9.3f}'.format(_s) for _s in (np.exp(r_lvar_b.flatten()))]))

    if opt[1] is not None:
        q_param = opt[1].target._dict['params']
        print('\tProposal')

        q_mean_w = q_param['head_mean_fn']['kernel']
        print('\t\tQ mean weight:           ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_mean_w.flatten()]))

        try:
            q_mean_b = q_param['head_mean_fn']['bias']  # ADD THIS BACK IF WE ARE USING THE BIAS
            print('\t\tQ mean bias       (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in (np.exp(q_mean_b.flatten()))]))
        except:
            pass

        try:
            q_lvar_w = q_param['head_log_var_fn']['kernel']
            print('\t\tQ var weight      (->0): ', '  '.join(['{: >9.3f}'.format(_s) for _s in q_lvar_w.flatten()]))
        except:
            pass

        q_lvar_b = q_param['head_log_var_fn']['bias']
        print('\t\tQ var bias:              ', '  '.join(['{: >9.3f}'.format(_s) for _s in (np.exp(q_lvar_b.flatten()))]))

    print()
    print()
    print()
