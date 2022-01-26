import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import argparse
from jax import random as jr
import flax.linen as nn
from typing import NamedTuple
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd
import pickle

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.vrnn.models import VRNN
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.fivo as fivo
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts


def get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()

    parser.add_argument('--PLOT', default=0, type=int)  # TODO - note always disable plotting on the VRNN example.

    parser.add_argument('--DATASET', default='pianorolls', type=str)  # TODO - only pianorolls is set up.
    parser.add_argument('--synthetic-data', default=0, type=int)

    parser.add_argument('--resampling-criterion', default='always_resample', type=str)  # CSV.  # {'always_resample', 'never_resample'}.

    parser.add_argument('--use-sgr', default=1, type=int)  # {0, 1}.

    parser.add_argument('--temper', default=0.0, type=float)  # {0.0 to disable,  >0.1 to temper}.

    # {'params_rnn', 'params_prior', 'params_decoder_latent', 'params_decoder_full', 'params_encoder_data'}.
    parser.add_argument('--free-parameters', default='params_rnn,params_prior,params_decoder_latent,params_decoder_full,params_encoder_data',
                        type=str)  # CSV.

    parser.add_argument('--proposal-structure', default='BOOTSTRAP', type=str)  # {None/'NONE'/'BOOTSTRAP', 'DIRECT', 'RESQ', }
    parser.add_argument('--proposal-type', default='SINGLE_WINDOW',
                        type=str)  # {PERSTEP_ALLOBS, 'PERSTEP_SINGLEOBS', 'SINGLE_SINGLEOBS', 'PERSTEP_WINDOW', 'SINGLE_WINDOW'}
    parser.add_argument('--proposal-window-length', default=2, type=int)  # {int, None}.
    parser.add_argument('--proposal-fn-family', default='AFFINE', type=str)  # {'AFFINE', 'MLP'}.

    parser.add_argument('--tilt-structure', default='NONE', type=str)  # {None/'NONE', 'DIRECT', 'VRNN'}
    parser.add_argument('--tilt-type', default='SINGLE_WINDOW', type=str)  # {'PERSTEP_ALLOBS', 'PERSTEP_WINDOW', 'SINGLE_WINDOW'}.
    parser.add_argument('--tilt-window-length', default=2, type=int)  # {int, None}.
    parser.add_argument('--tilt-fn-family', default='AFFINE', type=str)  # {'AFFINE', 'MLP'}.

    parser.add_argument('--num-particles', default=4, type=int)
    parser.add_argument('--datasets-per-batch', default=3, type=int)
    parser.add_argument('--opt-steps', default=100000, type=int)

    parser.add_argument('--lr-p', default=0.0001, type=float)
    parser.add_argument('--lr-q', default=0.0001, type=float)
    parser.add_argument('--lr-r', default=0.0001, type=float)

    parser.add_argument('--latent-dim', default=5, type=int)
    parser.add_argument('--emissions-dim', default=11, type=int)
    parser.add_argument('--latent-enc-dim', default=12, type=int)
    parser.add_argument('--obs-enc-dim', default=13, type=int)
    parser.add_argument('--rnn-state-dim', default=64, type=int)

    parser.add_argument('--T', default=29, type=int)  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.
    parser.add_argument('--num-trials', default=100000, type=int)
    parser.add_argument('--num-val-dataset-fraction', default=0.2, type=int)

    parser.add_argument('--dset-to-plot', default=2, type=int)
    parser.add_argument('--validation-particles', default=250, type=int)
    parser.add_argument('--sweep-test-particles', default=10, type=int)

    parser.add_argument('--load-path', default=None, type=str)  # './params_vrnn_tmp.p'
    parser.add_argument('--save-path', default=None, type=str)  # './params_vrnn_tmp.p'
    parser.add_argument('--model', default='VRNN', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--log-group', default='debug', type=str)  # {'debug', 'gdm-v1.0'}

    parser.add_argument('--vi-use-tilt-gradient', default=0, type=int)  # {0, 1}.
    parser.add_argument('--vi-buffer-length', default=10, type=int)  #
    parser.add_argument('--vi-minibatch-size', default=16, type=int)  #
    parser.add_argument('--vi-epochs', default=1, type=int)  #

    config = parser.parse_args().__dict__

    # Make sure this one is formatted correctly.
    config['model'] = 'VRNN'

    # Force the tilt temperature to zero if we are not using tilts.  this is just bookkeeping, really.
    if config['tilt_structure'] == 'NONE' or config['tilt_structure'] is None:
        config['temper'] = 0.0

    assert not config['vi_use_tilt_gradient'], "NO IDEA IF THIS WILL WORK YET..."
    assert len(config['free_parameters'].split(',')) == 5, "NOT OPTIMIZING ALL VRNN PARAMETERS..."

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
    true_model, true_states, dataset, dataset_masks = define_true_model_and_data(subkey, env)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = define_test_model(subkey, true_model, env)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = define_proposal(subkey, model, dataset, env)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = define_tilt(subkey, model, dataset, env)

    # Build up the train/val splits.
    num_val_datasets = int(len(dataset) * env.config.num_val_dataset_fraction)
    validation_datasets = dataset[:num_val_datasets]
    validation_dataset_masks = dataset_masks[:num_val_datasets]
    train_datasets = dataset[num_val_datasets:]
    train_dataset_masks = dataset_masks[num_val_datasets:]

    # Return this big pile of stuff.
    ret_model = (true_model, true_states, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    return ret_model, ret_test, ret_prop, ret_tilt


def define_tilt(subkey, model, dataset, env):
    """

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """

    if (env.config.tilt_structure is None) or (env.config.tilt_structure == 'NONE'):
        return None, None, lambda *args: None

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

    # trunk_fn = nn_util.MLP([6, ], output_layer_relu=True)
    # head_mean_fn = nn.Dense(dummy_tilt_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_tilt_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.01, )

    # Define the tilts themselves.
    tilt = tilt_fn(n_tilts=n_tilts,
                   tilt_input=stock_tilt_input,
                   trunk_fn=trunk_fn,
                   head_mean_fn=head_mean_fn,
                   head_log_var_fn=head_log_var_fn, )

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
        return None, None, lambda *args: None

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, ).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input = (dataset[0], model, dummy_particles, 0, dummy_p_dist,)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    # If we are using RESQ, define a kernel that basically does nothing to begin with.
    if env.config.proposal_structure == 'RESQ':
        kernel_init = lambda *args: nn.initializers.lecun_normal()(*args) * 0.1
    else:
        kernel_init = None

    trunk_fn = None
    head_mean_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=kernel_init)
    head_log_var_fn = nn_util.Static(dummy_proposal_output.shape[0], bias_init=nn.initializers.zeros)

    # trunk_fn = nn_util.MLP([6, ], output_layer_relu=True)
    # head_mean_fn = nn.Dense(dummy_proposal_output.shape[0])
    # head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0], kernel_init=lambda *args: nn.initializers.lecun_normal()(*args) * 0.01, )

    # configure the tilt.
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
        n_props = 1
        proposal_window_length = env.config.proposal_window_length

    else:
        raise NotImplementedError()

    # Define the proposal itself.
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
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, env.config.proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn


def load_piano_data():
    from ssm.inference.data.datasets import sparse_pianoroll_to_dense

    with open('./data/piano-midi.de.pkl', 'rb') as f:
        dataset_sparse = pickle.load(f)

    PAD_FLAG = 0.0
    MAX_LENGTH = 200

    min_note = 21
    max_note = 108
    num_notes = max_note - min_note + 1

    dataset_and_metadata = [sparse_pianoroll_to_dense(_d, min_note=min_note, num_notes=num_notes) for _d in dataset_sparse['train']]
    max_length = max([_d[1] for _d in dataset_and_metadata])

    if max_length < MAX_LENGTH:
        MAX_LENGTH = max_length

    dataset_masks = []
    dataset = []
    for _i, _d in enumerate(dataset_and_metadata):

        if len(_d[0]) > MAX_LENGTH:
            print('[WARNING]: Removing dataset, over length 1000.')
            continue

        dataset.append(np.concatenate((_d[0], PAD_FLAG * np.ones((MAX_LENGTH - len(_d[0]), *_d[0].shape[1:])))))
        dataset_masks.append(np.concatenate((np.ones(_d[0].shape[0]), 0.0 * np.ones((MAX_LENGTH - len(_d[0]))))))

    print('Loaded {} datasets.'.format(len(dataset)))

    dataset = np.asarray(dataset)
    dataset_masks = np.asarray(dataset_masks)
    true_states = None  # There are no true states!

    print('\n\nWARNING trimming data further. \n\n')
    dataset = dataset[:, :20]
    dataset_masks = dataset_masks[:, :20]

    dataset_means = dataset_sparse['train_mean']

    return dataset, dataset_masks, true_states, dataset_means


def define_true_model_and_data(key, env):
    """

    Args:
        key:

    Returns:

    """

    RNN_CELL_TYPE = nn.LSTMCell

    # Pull out the sizes of things,
    latent_dim = env.config.latent_dim
    latent_encoded_dim = env.config.latent_enc_dim
    emissions_encoded_dim = env.config.obs_enc_dim
    rnn_state_dim = env.config.rnn_state_dim

    if env.config.synthetic_data:
        emissions_dim = env.config.emissions_dim
        dataset_means = np.zeros((emissions_dim,))
        output_type = 'GAUSSIAN'

        num_trials = env.config.num_trials
        T = env.config.T  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.
        dataset_masks = np.ones((num_trials, T + 1,))

    else:

        if env.config.DATASET == 'pianorolls':
            dataset, dataset_masks, true_states, dataset_means = load_piano_data()
            emissions_dim = dataset.shape[-1]
            output_type = 'BERNOULLI'
        else:
            raise NotImplementedError()

    kernel_init = lambda *args: nn.initializers.xavier_normal()(*args)  # * 0.001

    # Define some dummy place holders.
    val_obs = np.zeros(emissions_dim)
    val_latent = np.zeros(latent_dim)
    val_latent_decoded = np.zeros(latent_encoded_dim)
    val_data_encoded = np.zeros(emissions_encoded_dim)

    # Build up each of the functions.
    prior = nn.Dense(2 * latent_dim, kernel_init=kernel_init)
    latent_decoder = nn.Dense(latent_encoded_dim, kernel_init=kernel_init)
    data_encoder = nn.Dense(emissions_encoded_dim, kernel_init=kernel_init)

    # We have to do something a bit funny to the full decoder to make sure that it is approximately correct.
    output_kernel_init = lambda *_args: nn.initializers.lecun_normal()(*_args) * 0.1
    output_bias_init = lambda *_args: np.log(dataset_means)

    if output_type == 'BERNOULLI':
        full_decoder = nn.Dense(emissions_dim, kernel_init=output_kernel_init, bias_init=output_bias_init)
    elif output_type == 'GAUSSIAN':
        full_decoder = nn.Dense(2 * emissions_dim, kernel_init=output_kernel_init, bias_init=output_bias_init)
    else:
        raise NotImplementedError()

    # generate the keys for initialization.
    key, *subkeys = jr.split(key, num=7)

    # The RNN requires a little more of a delicate touch.
    rnn = RNN_CELL_TYPE()
    rnn_carry = rnn.initialize_carry(subkeys[0], batch_dims=(), size=rnn_state_dim)
    input_rnn = tuple((rnn_carry, np.concatenate((val_latent_decoded, val_data_encoded))))
    params_rnn = rnn.init(subkeys[1], input_rnn[0], input_rnn[1])

    # Work out the dimensions of the inputs to each function.
    input_latent_decoder = val_latent
    input_data_encoder = val_obs
    input_full_decoder = np.concatenate((rnn_carry[1], val_latent_decoded))
    input_prior = rnn_carry[1]

    # Initialize the functions and grab the parameters.
    params_prior = prior.init(subkeys[2], input_prior)
    params_latent_decoder = latent_decoder.init(subkeys[3], input_latent_decoder)
    params_full_decoder = full_decoder.init(subkeys[4], input_full_decoder)
    params_data_encoder = data_encoder.init(subkeys[5], input_data_encoder)

    # Define functions that are essentially the "from param".
    rebuild_rnn = lambda _params: rnn.bind(_params)
    rebuild_prior = lambda _params: prior.bind(_params)
    rebuild_latent_decoder = lambda _params: latent_decoder.bind(_params)
    rebuild_full_decoder = lambda _params: full_decoder.bind(_params)
    rebuild_data_encoder = lambda _params: data_encoder.bind(_params)

    # Create the true model.
    key, subkey = jr.split(key)
    true_model = VRNN(num_latent_dims=latent_dim,
                      num_emission_dims=emissions_dim,
                      latent_enc_dim=latent_encoded_dim,
                      obs_enc_dim=emissions_encoded_dim,
                      output_type=output_type,
                      rnn_state_dim=rnn_state_dim,
                      rebuild_rnn=rebuild_rnn,
                      rebuild_prior=rebuild_prior,
                      rebuild_decoder_latent=rebuild_latent_decoder,
                      rebuild_decoder_full=rebuild_full_decoder,
                      rebuild_encoder_data=rebuild_data_encoder,
                      params_rnn=params_rnn,
                      params_prior=params_prior,
                      params_decoder_latent=params_latent_decoder,
                      params_decoder_full=params_full_decoder,
                      params_encoder_data=params_data_encoder, )

    # Load up the data first to pull out the dimensions.
    if env.config.synthetic_data:
        # Sample some data.  NOTE - we use the unconditional sample subroutine here to generate data and obs.
        key, subkey = jr.split(key)
        true_states, dataset = true_model.unconditional_sample(key=subkey, num_steps=T + 1, num_samples=num_trials)
        dataset_masks = np.ones(dataset.shape[0])

    return true_model, true_states, dataset, dataset_masks


def define_test_model(key, true_model, env):
    """

    Args:
        key:
        true_model:
        env:

    Returns:

    """
    key, subkey = jr.split(key)

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, env.config.free_parameters)

    if len(env.config.free_parameters) > 0:

        print('\n\n\n[WARNING]: This way of initializing parameters is crap.\n\n\n')

        # Get the default parameters from the true model.
        true_params = fivo.get_model_params_fn(true_model)

        # Generate a model to use.  NOTE - this will generate a new model, and we will
        # overwrite any of the free parameters of interest into the true model.
        # TODO - NOTE - this clones the true model.  need to double-check that the parameters are deeply mutated later.
        tmp_model = true_model.__class__(*true_params)

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # Mutate the free parameters.
        for _k in env.config.free_parameters:
            print('[WARNING]: No initializer for {}.'.format(_k))
            # _base = getattr(default_params, _k)
            # key, subkey = jr.split(key)
            #
            # # TODO - This needs to be made model-specific.
            #
            # # TODO - this initialization method is crap.
            # new_val = {_k: _base + (0.1 * jr.normal(key=subkey, shape=_base.shape))}
            #
            # default_params = utils.mutate_named_tuple_by_key(default_params, new_val)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params, param_figs):
    """
    TODO - essentially not plotting here.

    Args:
        _param_hist:
        _loss_hist:
        _true_loss_em:
        _true_loss_smc:
        _true_params:
        param_figs:

    Returns:

    """
    return param_figs


def do_print(_step, true_model, opt, true_lml, pred_lml, pred_fivo_bound, em_log_marginal_likelihood=None):
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
    _str = 'Step: {:> 7d},  True Neg-LML: {:> 8.3f},  Pred Neg-LML: {:> 8.3f},  Pred FIVO bound {:> 8.3f}'. \
        format(_step, true_lml, pred_lml, pred_fivo_bound)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {:> 8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    if opt[0] is not None:
        if len(opt[0].target) > 0:
            # print()
            print('\tModel:  Good luck printing that shite.')

    print()
    print()
    print()


def get_true_target_marginal(model, data):
    """
    Take in a model and some data and return the tfd distribution representing the marginals of true posterior.
    Args:
        model:
        data:

    Returns:

    """
    return None