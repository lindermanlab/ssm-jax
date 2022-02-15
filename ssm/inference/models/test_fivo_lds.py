import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import argparse
import flax.linen as nn
from jax import random as jr
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
from ssm.inference.fivo_util import pretrain_encoder
import ssm.inference.encoders as encoders


def get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()

    # General sweep settings.
    parser.add_argument('--validation-interval', default=2000, type=int)
    parser.add_argument('--train-resampling-criterion', default='ess_criterion', type=str)  # {'always_resample', 'never_resample', 'ess_criterion'}.
    parser.add_argument('--eval-resampling-criterion', default='ess_criterion', type=str)  # {'always_resample', 'never_resample', 'ess_criterion'}.
    parser.add_argument('--resampling-function', default='multinomial_resampling', type=str)  # CSV.  # {'multinomial_resampling', 'systematic_resampling'}.
    parser.add_argument('--use-sgr', default=1, type=int)  # {0, 1}
    parser.add_argument('--temper', default=0.0, type=float)  # {0.0 to disable,  >0.1 to temper}.
    parser.add_argument('--num-particles', default=4, type=int)
    parser.add_argument('--datasets-per-batch', default=8, type=int)

    # Model free parameters.
    parser.add_argument('--free-parameters', default='dynamics_weights', type=str)  # CSV.  # {'dynamics_bias', 'dynamics_weights'}.

    # Proposal args.
    parser.add_argument('--proposal-structure', default='RESQ', type=str)  # {None/'NONE'/'BOOTSTRAP', 'DIRECT', 'RESQ', }
    parser.add_argument('--proposal-type', default='SINGLE_WINDOW', type=str)  # {PERSTEP_ALLOBS, 'PERSTEP_SINGLEOBS', 'SINGLE_SINGLEOBS', 'PERSTEP_WINDOW', 'SINGLE_WINDOW'}
    parser.add_argument('--proposal-window-length', default=2, type=int)             # {int, None}.
    parser.add_argument('--proposal-fn-family', default='MLP', type=str)         # {'AFFINE', 'MLP'}.

    # Tilt args.
    parser.add_argument('--tilt-structure', default='DIRECT', type=str)  # {None/'NONE', 'DIRECT'}
    parser.add_argument('--tilt-type', default='SINGLE_WINDOW', type=str)  # {'PERSTEP_ALLOBS', 'PERSTEP_WINDOW', 'SINGLE_WINDOW'}.
    parser.add_argument('--tilt-window-length', default=2, type=int)  # {int, None}.
    parser.add_argument('--tilt-fn-family', default='AFFINE', type=str)             # {'AFFINE', 'MLP'}.

    # Variational / M.L. learning of tilt function.
    parser.add_argument('--vi-use-tilt-gradient', default=0, type=int)  # {0, 1}.
    parser.add_argument('--vi-buffer-length', default=10, type=int)  #
    parser.add_argument('--vi-minibatch-size', default=16, type=int)  #
    parser.add_argument('--vi-epochs', default=1, type=int)  #

    # Learning rates.
    parser.add_argument('--lr-p', default=0.001, type=float)
    parser.add_argument('--lr-q', default=0.001, type=float)
    parser.add_argument('--lr-r', default=0.001, type=float)
    parser.add_argument('--lr-e', default=3.0e-5, type=float, help="Learning rate of data encoder parameters.")

    # Architecture/data args.
    parser.add_argument('--T', default=29, type=int)   # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.
    parser.add_argument('--latent-dim', default=1, type=int)
    parser.add_argument('--emissions-dim', default=1, type=int)
    parser.add_argument('--num-trials', default=100000, type=int)
    parser.add_argument('--num-val-datasets', default=100, type=int)

    parser.add_argument('--opt-steps', default=100000, type=int)
    parser.add_argument('--dset-to-plot', default=2, type=int)
    parser.add_argument('--validation-particles', default=128, type=int)
    parser.add_argument('--sweep-test-particles', default=4, type=int)
    parser.add_argument('--load-path', default=None, type=str)  # './params_lds_tmp.p'
    parser.add_argument('--save-path', default=None, type=str)  # './params_lds_tmp.p'
    parser.add_argument('--model', default='LDS', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log-group', default='debug-lds', type=str)  # {'debug', 'gdm-v1.0'}
    parser.add_argument('--plot-interval', default=1, type=int)
    parser.add_argument('--log-to-wandb-interval', default=1, type=int)
    parser.add_argument('--PLOT', default=1, type=int)
    parser.add_argument('--encoder-structure', default='NONE', type=str)  # {None/'NONE', 'BIRNN' }
    parser.add_argument('--dataset', default='default', type=str)
    parser.add_argument('--synthetic-data', default=1, type=int)
    parser.add_argument('--use-bootstrap-initial-distribution', default=0, type=int, help="Force sweeps to use the model for initialization.")
    parser.add_argument('--l2-reg', default=0.000, type=float, help="L2 regulation hyperparameter.")

    config = parser.parse_args().__dict__

    # Make sure this one is formatted correctly.
    config['model'] = 'LDS'

    return config, do_print, define_test, do_plot, get_true_target_marginal


def define_test(key, env):
    """

    Args:
        key:
        env:

    Returns:

    """

    # Define the true model.
    key, subkey = jr.split(key)
    true_model, true_states, datasets, masks = define_true_model_and_data(subkey, env)

    if len(datasets.shape) == 2:
        print('\nWARNING: Expanding dataset and mask dim.\n')
        datasets = np.expand_dims(datasets, 0)
        masks = np.expand_dims(masks, 0)

    val_datasets = np.asarray(dc(datasets[:env.config.num_val_datasets]))
    val_dataset_masks = np.asarray(dc(masks[:env.config.num_val_datasets]))
    tst_datasets = np.asarray(dc(datasets[env.config.num_val_datasets:(2 * env.config.num_val_datasets)]))
    tst_dataset_masks = np.asarray(dc(masks[env.config.num_val_datasets:(2 * env.config.num_val_datasets)]))
    trn_datasets = np.asarray(dc(datasets[(2 * env.config.num_val_datasets):]))
    trn_dataset_masks = np.asarray(dc(masks[(2 * env.config.num_val_datasets):]))

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = define_test_model(subkey, true_model, env)

    # Define an encoder for the data.
    key, subkey = jax.random.split(key)
    encoder, encoder_params, rebuild_encoder_fn = define_data_encoder(subkey, true_model, env,
                                                                      trn_datasets, trn_dataset_masks,
                                                                      val_datasets, val_dataset_masks)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = define_proposal(subkey, model, datasets, env)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = define_tilt(subkey, model, datasets, env)

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, trn_datasets, trn_dataset_masks, val_datasets, val_dataset_masks, tst_datasets, tst_dataset_masks)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    ret_enc = (encoder, encoder_params, rebuild_encoder_fn)
    return ret_model, ret_test, ret_prop, ret_tilt, ret_enc


def define_data_encoder(key, true_model, env, trn_datasets, trn_dataset_masks, val_datasets, val_dataset_masks):
    """

    Args:
        subkey:
        true_model:
        env:
        trn_datasets:
        trn_dataset_masks:

    Returns:

    """

    # If there is no encoder, just pass nones through.
    if (env.config.encoder_structure == 'NONE') or (env.config.encoder_structure is None):
        return None, None, lambda *_args: None

    key, subkey1, subkey2 = jr.split(key, num=3)

    # # Define the reccurent bit.
    # rnn_state_dim = env.config.rnn_state_dim if env.config.rnn_state_dim is not None else env.config.latent_dim
    # rnn_obj = nn_util.RnnWithReadoutLayer(trn_datasets.shape[-1], rnn_state_dim)
    rnn_obj = nn.LSTMCell()

    data_encoder = encoders.IndependentBiRnnEncoder(env, subkey1, rnn_obj, trn_datasets[0])

    if env.config.encoder_pretrain:
        encoder_params = pretrain_encoder(env, subkey2, data_encoder,
                                          trn_datasets, trn_dataset_masks, val_datasets, val_dataset_masks)
    else:
        subkey1, subkey2 = jr.split(key)
        init_carry = data_encoder.initialize_carry(subkey1)
        encoder_params = data_encoder.init(subkey2, (init_carry, np.zeros(trn_datasets.shape[-1])))

    rebuild_encoder_fn = encoders.rebuild_encoder(data_encoder, env)

    return data_encoder, encoder_params, rebuild_encoder_fn


def define_tilt(subkey, model, dataset, env):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if (env.config.tilt_structure is None) or (env.config.tilt_structure == 'NONE'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # configure the tilt.
    if env.config.tilt_type == 'PERSTEP_ALLOBS':
        tilt_fn = tilts.IGPerStepTilt
        n_tilts = len(dataset[0]) - 1
        tilt_window_length = None

    elif env.config.tilt_type == 'PERSTEP_WINDOW':
        tilt_fn = tilts.IGWindowTilt
        n_tilts = len(dataset[0]) - 1
        tilt_window_length = env.config.tilt_window_length

    elif env.config.tilt_type == 'SINGLE_WINDOW':
        tilt_fn = tilts.IGWindowTilt
        n_tilts = 1
        tilt_window_length = env.config.tilt_window_length

    else:
        raise NotImplementedError()

    tilt_inputs = ()

    # Tilt functions take in (dataset, model, particles, t-1).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )

    # Generate the inputs.
    stock_tilt_input = (dataset[-1], model, dummy_particles[0], 0, tilt_window_length, *tilt_inputs)

    # Generate the outputs.
    dummy_tilt_output = tilt_fn._tilt_output_generator(*stock_tilt_input)

    # Define any custom link functions.
    trunk_fn = None
    head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    head_log_var_fn = nn_util.Static(dummy_tilt_output.shape[0])

    # trunk_fn = nn_util.MLP([6, ], output_layer_activation=True)
    # head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.01, )

    # Define the tilts themselves.
    print('Defining {} tilts.'.format(n_tilts))
    tilt = tilt_fn(n_tilts=n_tilts,
                   tilt_input=stock_tilt_input,
                   trunk_fn=trunk_fn,
                   head_mean_fn=head_mean_fn,
                   head_log_var_fn=head_log_var_fn,)

    # Initialize the network.
    tilt_params = tilt.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_tilt_fn = tilts.rebuild_tilt(tilt, env)
    return tilt, tilt_params, rebuild_tilt_fn


def define_proposal(subkey, model, dataset, env):
    """

    Args:
        subkey:
        model:
        dataset:
        env:

    Returns:

    """

    if env.config.proposal_structure in [None, 'NONE', 'BOOTSTRAP']:
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    dummy_q_state = None
    stock_proposal_input = (dataset[0], model, dummy_particles, 0, dummy_p_dist, dummy_q_state)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    # Configure the proposal structure.
    if env.config.proposal_type == 'PERSTEP_ALLOBS':
        proposal_cls = proposals.IndependentGaussianProposal
        n_props = len(dataset[0])
        proposal_window_length = None

    elif env.config.proposal_type == 'PERSTEP_SINGLEOBS':
        proposal_cls = proposals.IGSingleObsProposal
        n_props = len(dataset[0])
        proposal_window_length = None

    elif env.config.proposal_type == 'SINGLE_SINGLEOBS':
        proposal_cls = proposals.IGSingleObsProposal
        n_props = 1
        proposal_window_length = None

        # TODO - test this.
        raise NotImplementedError()

    elif env.config.proposal_type == 'PERSTEP_WINDOW':
        proposal_cls = proposals.IGWindowProposal
        n_props = len(dataset[0])
        proposal_window_length = env.config.proposal_window_length

    elif env.config.proposal_type == 'SINGLE_WINDOW':
        proposal_cls = proposals.IGWindowProposal

        if env.config.proposal_fn_family == 'MLP':
            n_props = 1  # If we are using an MLP, it should be able to handle this.
        elif env.config.proposal_fn_family == 'AFFINE':
            n_props = 2  # NOTE - 2 specifies an initial proposal and then a single proposal therein.
        proposal_window_length = env.config.proposal_window_length

    else:
        raise NotImplementedError()

    # If we are using RESQ, define a kernel that basically does nothing to begin with.
    if env.config.proposal_structure == 'RESQ':
        kernel_init = lambda *args: nn.initializers.lecun_normal()(*args) * 0.1
    else:
        kernel_init = None

    # Fork on the specified definition of the proposal.
    if env.config.proposal_fn_family == 'AFFINE':
        trunk_fn = None
        head_mean_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=kernel_init)
        head_log_var_fn = nn_util.Static(dummy_proposal_output.shape[0], bias_init=nn.initializers.zeros)

    elif env.config.proposal_fn_family == 'MLP':
        trunk_fn = nn_util.MLP([10, 10, ], output_layer_activation=True)
        head_mean_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=kernel_init)
        head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.1, )

    else:
        raise NotImplementedError()

    # Define the proposal itself.
    print('Defining {} proposals.'.format(n_props))
    proposal = proposal_cls(n_proposals=n_props,
                            stock_proposal_input=stock_proposal_input,
                            dummy_output=dummy_proposal_output,
                            trunk_fn=trunk_fn,
                            head_mean_fn=head_mean_fn,
                            head_log_var_fn=head_log_var_fn,
                            proposal_window_length=proposal_window_length)

    # Initialize the network.
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, env)
    return proposal, proposal_params, rebuild_prop_fn


def get_true_target_marginal(model, data):
    """
    Take in a model and some data and return the tfd distribution representing the marginals of true posterior.
    Args:
        model:
        data:

    Returns:

    """

    try:
        pred_em_posterior = jax.vmap(model.e_step)(data)

        marginal_mean = pred_em_posterior.mean().squeeze()
        marginal_std = np.sqrt(pred_em_posterior.covariance().squeeze())

        pred_em_marginal = tfd.MultivariateNormalDiag(marginal_mean, marginal_std)
    except:
        pred_em_marginal = None

    return pred_em_marginal


def define_true_model_and_data(key, env):
    """

    Args:
        key:

    Returns:

    """

    # dynamics_scale_tril = None
    # emission_scale_tril = None
    # initial_state_scale_tril = None
    true_dynamics_weights = None
    true_emission_weights = None

    latent_dim = env.config.latent_dim
    emissions_dim = env.config.emissions_dim
    num_trials = env.config.num_trials
    T = env.config.T  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.

    # If we are in the 1-D case, then we need to define a boring LDS.
    if latent_dim == 1:

        # NOTE - Set the dynamics scale here.
        # This needs to be done for all models because it is defined poorly inside the model.
        dynamics_scale_tril = 1.0 * np.eye(latent_dim)

        # NOTE - can make observations tighter here.
        emission_scale_tril = 1.0 * np.eye(emissions_dim)

        # NOTE - change the initial scale here.
        initial_state_scale_tril = 1.0 * np.eye(latent_dim)

        # Set up the transmission and emission weights to be unity.
        true_dynamics_weights = np.eye(latent_dim)
        true_emission_weights = np.eye(emissions_dim, latent_dim)

    else:

        # raise NotImplementedError('Still need to work on these settings a btt.')

        # NOTE - Set the dynamics scale here.
        # This needs to be done for all models because it is defined poorly inside the model.
        dynamics_scale_tril = 0.1 * np.eye(latent_dim)

        # NOTE - can make observations tighter here.
        emission_scale_tril = 1.0 * np.eye(emissions_dim)

        # NOTE - change the initial scale here.
        initial_state_scale_tril = 1.0 * np.eye(latent_dim)

    # Create the true model.
    key, subkey = jr.split(key)
    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=dynamics_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_weights=true_emission_weights,
                             emission_scale_tril=emission_scale_tril,
                             initial_state_scale_tril=initial_state_scale_tril
                             )

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, datasets = true_model.sample(key=subkey, num_steps=T+1, num_samples=num_trials)

    # Set up the mask as all visible.
    masks = np.ones((num_trials, T+1))

    return true_model, true_states, datasets, masks


def define_test_model(key, true_model, env):
    """

    Args:
        key:
        true_model:
        free_parameters:

    Returns:

    """
    key, subkey = jr.split(key)

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, env.config.free_parameters)

    if len(env.config.free_parameters) > 0:

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
        for _k in env.config.free_parameters:
            _base = getattr(default_params, _k)
            key, subkey = jr.split(key)

            # TODO - This needs to be made model-specific.
            if _k == 'dynamics_bias':
                new_val = {_k: _base + ((2.0 * jr.uniform(key=subkey, shape=_base.shape)) - 1.0)}
            elif _k == 'dynamics_weights':
                new_val = {_k: _base * (0.5 + jr.uniform(key=subkey, shape=_base.shape))}
            else:
                raise NotImplementedError()

            default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params,
                param_figs):
    """
    NOTE - removed proposal and tilt parameters here as they will be too complex.

    Args:
        _param_hist:
        _loss_hist:
        _true_loss_em:
        _true_loss_smc:
        _true_params:
        param_figs:

    Returns:

    """

    fsize = (12, 8)
    idx_to_str = lambda _idx: ['Model (p): '][_idx]

    for _p, _hist in enumerate(_param_hist[:1]):

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

                plt.savefig('./figs/lds_param_{}.pdf'.format(_p))

    return param_figs


def do_print(_step, true_model, opt, true_lml, true_fivo, pred_lml, pred_fivo_bound, em_log_marginal_likelihood, smoothed_training_loss):
    """

    Args:
        _step:
        true_model:
        opt:
        true_lml:
        pred_lml:
        pred_fivo_bound:
        em_log_marginal_likelihood:

    Returns:

    """
    _str = 'Step: {:> 5d},  '.format(_step) + \
           'Smoothed training loss: {:> 8.3f},   '.format(smoothed_training_loss) + \
           'Val EM Neg-LML: {:> 8.3f},  '.format(em_log_marginal_likelihood) + \
           'Val True Neg-LML: {:> 8.3f},  '.format(true_lml) + \
           'Val Pred Neg-LML: {:> 8.3f},  '.format(pred_lml) + \
           'Val True neg FIVO bound: {:> 8.3f},  '.format(true_fivo) + \
           'Val Pred neg FIVO bound: {:> 8.3f},'.format(pred_fivo_bound)
    print(_str)

    if opt[0] is not None:
        if len(opt[0].target) > 0:
            # print()
            print('\tModel')

            true_str = [_k + ': ' + ' '.join(['{:> 5.3f}'.format(_f) for _f in getattr(true_model._parameters, _k).flatten()]) for _k in opt[0].target._fields]
            pred_str = [_k + ': ' + ' '.join(['{:> 5.3f}'.format(_f) for _f in getattr(opt[0].target, _k).flatten()]) for _k in opt[0].target._fields]

            print('\t\tTrue: ' + str(true_str))
            print('\t\tPred: ' + str(pred_str))

    # NOTE - the proposal and tilt are more complex here, so don't show them.

    print()
    print()
    print()
