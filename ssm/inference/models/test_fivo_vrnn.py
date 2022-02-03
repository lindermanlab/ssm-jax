import jax
import argparse
import tensorflow as tf
import flax.linen as nn
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random as jr
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd
from flax import optim

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.vrnn.models import VRNN, VrnnFilteringProposal, VrnnSmoothingProposal
import ssm.nn_util as nn_util
import ssm.inference.fivo as fivo
from ssm.inference.fivo_util import pretrain_encoder
import ssm.inference.proposals as proposals
import ssm.inference.tilts as tilts
import ssm.inference.encoders as encoders
import ssm.inference.fivo_util as fivo_util


def get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-interval', default=50, type=int)

    parser.add_argument('--encoder-structure', default='BIRNN', type=str)  # {None/'NONE', 'BIRNN' }

    parser.add_argument('--dataset', default='jsb.pkl', type=str,
                        help="Dataset to apply method to.  {'piano-midi.pkl', 'nottingham.pkl', 'musedata.pkl', 'jsb.pkl'}. ")

    parser.add_argument('--resampling-criterion', default='always_resample', type=str)  # {'always_resample', 'never_resample'}.
    parser.add_argument('--resampling-function', default='multinomial_resampling', type=str)  # {'multinomial_resampling', 'systematic_resampling'}.
    parser.add_argument('--use-sgr', default=1, type=int, help="{0, 1}.")
    parser.add_argument('--temper', default=0.0, type=float, help="{0.0 to disable,  >0.1 to temper}")

    # CSV: {'params_rnn', 'params_prior', 'params_decoder_latent', 'params_decoder_full', 'params_encoder_data'}.
    parser.add_argument('--free-parameters', default='params_rnn,params_prior,params_decoder_latent,params_decoder_full,params_encoder_data',type=str)

    parser.add_argument('--proposal-structure', default='VRNN_FILTERING_RESQ', type=str)  # {None/'NONE'/'BOOTSTRAP', 'VRNN_FILTERING_RESQ', 'VRNN_SMOOTHING_RESQ' }
    parser.add_argument('--proposal-type', default='VRNN_FILTERING', type=str)  # {'VRNN_FILTERING', 'VRNN_SMOOTHING'}
    parser.add_argument('--proposal-window-length', default=1, type=int)  # {None, }.
    parser.add_argument('--proposal-fn-family', default='MLP', type=str)  # {'MLP', }.

    parser.add_argument('--tilt-structure', default='NONE', type=str)  # {None/'NONE', 'DIRECT', 'VRNN'}
    parser.add_argument('--tilt-type', default='SINGLE_WINDOW', type=str)  # {'PERSTEP_ALLOBS', 'PERSTEP_WINDOW', 'SINGLE_WINDOW'}.
    parser.add_argument('--tilt-window-length', default=2, type=int)  # {None, }.
    parser.add_argument('--tilt-fn-family', default='VRNN', type=str)  # {'VRNN'}.

    parser.add_argument('--num-particles', default=4, type=int, help="Number of particles per sweep during learning.")
    parser.add_argument('--datasets-per-batch', default=4, type=int, help="Number of datasets averaged across per FIVO step.")

    # VRNN architecture args.
    parser.add_argument('--latent-dim', default=32, type=int, help="Dimension of z latent variable.")
    parser.add_argument('--latent-enc-dim', default=None, type=int, help="Dimension of encoded latent z variable. (None -> latent_dim)")
    parser.add_argument('--obs-enc-dim', default=None, type=int, help="Dimension of encoded observations. (None -> latent_dim)")
    parser.add_argument('--rnn-state-dim', default=None, type=int, help="Dimension of the deterministic RNN. (None -> latent_dim)")
    parser.add_argument('--fcnet-hidden-sizes', default=None, type=str,
                        help="Layer widths of MLPs. CSV of widths, i.e. '10,10'. (None -> [latent_dim]). ")

    parser.add_argument('--lr-p', default=3.0e-5, type=float, help="Learning rate of model parameters.")
    parser.add_argument('--lr-q', default=3.0e-5, type=float, help="Learning rate of proposal parameters.")
    parser.add_argument('--lr-r', default=3.0e-5, type=float, help="Learning rate of tilt parameters.")
    parser.add_argument('--lr-e', default=3.0e-5, type=float, help="Learning rate of data encoder parameters.")

    parser.add_argument('--opt-steps', default=100000, type=int, help="Number of FIVO steps to take.")
    parser.add_argument('--dset-to-plot', default=2, type=int, help="Index of dataset to visualize.")
    parser.add_argument('--validation-particles', default=128, type=int, help="'Large' number of particles for asymptotic evaluation.")
    parser.add_argument('--sweep-test-particles', default=10, type=int, help="'Small' number of particles for finite-particle evalaution.")
    parser.add_argument('--load-path', default=None, type=str, help="File path to load model from.")  # './params_vrnn_tmp.p'
    parser.add_argument('--save-path', default=None, type=str, help="File path to save model to.")  # './params_vrnn_tmp.p'
    parser.add_argument('--model', default='VRNN', type=str, help="Don't change here.")
    parser.add_argument('--seed', default=10, type=int, help="Seed for initialization.")
    parser.add_argument('--log-group', default='debug-vrnn', type=str, help="WandB group to log to.  Overwrite from outside.")
    parser.add_argument('--plot-interval', default=1, type=int, help="Multiples of --validation-interval to plot at.")
    parser.add_argument('--log-to-wandb-interval', default=1, type=int, help="Multiples of --validation-interval to push to WandB remote at.")
    parser.add_argument('--PLOT', default=0, type=int, help="Whether to make plots online.  Always disable plotting for the VRNN.")
    parser.add_argument('--synthetic-data', default=0, type=int, help="Generate and use synthetic data for testing/debugging.")
    parser.add_argument('--use-bootstrap-initial-distribution', default=1, type=int, help="Force sweeps to use the model for initialization.")

    # Old stuff.
    parser.add_argument('--T', default=10, type=int, help="Length of sequences.  (Overwritten for real data)")
    parser.add_argument('--emissions-dim', default=1, type=int, help="Dimension of observed value (Overwritten for real data)")
    parser.add_argument('--num-trials', default=100000, type=int, help="Number of datasets to generate.  (Overwritten for real data)")
    parser.add_argument('--num-val-datasets', default=100, type=int, help="(Overwritten for real data)")

    parser.add_argument('--encoder-pretrain', default=0, type=int, help="{0, 1}")
    parser.add_argument('--encoder-pretrain-opt-steps', default=100, type=int, help="")
    parser.add_argument('--encoder-pretrain-lr', default=0.01, type=float, help="")
    parser.add_argument('--encoder-pretrain-batch-size', default=4, type=float, help="")

    parser.add_argument('--vi-use-tilt-gradient', default=0, type=int, help="Learn tilt using VI.")
    parser.add_argument('--vi-buffer-length', default=10, type=int, help="Number of optimization steps' data to store as VI buffer.  Linked to --batch-size.")
    parser.add_argument('--vi-minibatch-size', default=16, type=int, help="Size of VI minibatches when learning tilt with VI.")
    parser.add_argument('--vi-epochs', default=1, type=int, help="Number of VI epochs to perform when learning tilt with VI.")

    config = parser.parse_args().__dict__

    # Make sure this one is formatted correctly.
    config['model'] = 'VRNN'

    assert not config['vi_use_tilt_gradient'], "NO IDEA IF THIS WILL WORK YET..."
    assert len(config['free_parameters'].split(',')) == 5, "NOT OPTIMIZING ALL VRNN PARAMETERS..."

    # Quickly do some type conversion here.
    # Determine the size of the MLP link functions.
    if config['fcnet_hidden_sizes'] is None:
        config['fcnet_hidden_sizes'] = [config['latent_dim']]
    else:
        if type(config['fcnet_hidden_sizes']) == str:
            config['fcnet_hidden_sizes'] = tuple(int(_s) for _s in ('10,10'.replace(' ', '').split(',')))
        else:
            try:
                np.zeros(config['fcnet_hidden_sizes'])
            except:
                raise RuntimeError("Invalid size specification.")

    # If there is no RNN state dim specified, then just copy the latent dim.
    if config['rnn_state_dim'] is None:
        config['rnn_state_dim'] = config['latent_dim']

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
    true_model, train_true_states, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks = \
        define_true_model_and_data(subkey, env)

    # Now define a model to test.
    key, subkey = jax.random.split(key)
    model, get_model_params, rebuild_model_fn = define_test_model(subkey, true_model, env)

    # Define an encoder for the data.
    key, subkey = jax.random.split(key)
    encoder, encoder_params, rebuild_encoder_fn = define_data_encoder(subkey, true_model, env,
                                                                      train_datasets, train_dataset_masks,
                                                                      validation_datasets, validation_dataset_masks)

    # Define the proposal.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = define_proposal(subkey, model, train_datasets, env, data_encoder=encoder)

    # Define the tilt.
    key, subkey = jr.split(key)
    tilt, tilt_params, rebuild_tilt_fn = define_tilt(subkey, model, train_datasets, env)

    # Return this big pile of stuff.
    ret_model = (true_model, train_true_states, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks)
    ret_test = (model, get_model_params, rebuild_model_fn)
    ret_prop = (proposal, proposal_params, rebuild_prop_fn)
    ret_tilt = (tilt, tilt_params, rebuild_tilt_fn)
    ret_enc = (encoder, encoder_params, rebuild_encoder_fn)
    return ret_model, ret_test, ret_prop, ret_tilt, ret_enc


def define_data_encoder(key, true_model, env, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks):
    """

    Args:
        subkey:
        true_model:
        env:
        train_datasets:
        train_dataset_masks:

    Returns:

    """

    # If there is no encoder, just pass nones through.
    if (env.config.encoder_structure == 'NONE') or (env.config.encoder_structure is None):
        return None, None, lambda *_args: None

    key, subkey1, subkey2 = jr.split(key, num=3)

    # # Define the reccurent bit.
    # rnn_state_dim = env.config.rnn_state_dim if env.config.rnn_state_dim is not None else env.config.latent_dim
    # rnn_obj = nn_util.RnnWithReadoutLayer(train_datasets.shape[-1], rnn_state_dim)
    rnn_obj = nn.LSTMCell()

    data_encoder = encoders.IndependentBiRnnEncoder(env, subkey1, rnn_obj, train_datasets[0])

    if env.config.encoder_pretrain:
        encoder_params = pretrain_encoder(env, subkey2, data_encoder,
                                          train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks)
    else:
        subkey1, subkey2 = jr.split(key)
        init_carry = data_encoder.initialize_carry(subkey1)
        encoder_params = data_encoder.init(subkey2, (init_carry, np.zeros(train_datasets.shape[-1])))

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
        return None, None, lambda *args: None

    # configure the tilt.
    if env.config.tilt_type == 'SINGLE_WINDOW':
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
    print('Defining {} tilts.'.format(n_tilts))
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


def define_proposal(subkey, model, train_dataset, env, data_encoder=None):
    """

    Args:
        subkey:
        model:
        dataset:
        env:

    Returns:

    """
    subkey, subkey1, subkey2, subkey3 = jr.split(subkey, num=4)

    if env.config.proposal_structure in [None, 'NONE', 'BOOTSTRAP']:
        return None, None, lambda *args: None

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, ).
    n_dummy_particles = 4
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(n_dummy_particles, ), )
    dummy_obs = np.repeat(np.expand_dims(train_dataset[0, 0], 0), n_dummy_particles, axis=0)
    dummy_p_dist = model.dynamics_distribution(dummy_particles, covariates=(dummy_obs, ))
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )

    # Fork on the specified definition of the proposal.
    if env.config.proposal_fn_family == 'MLP':

        fcnet_hidden_sizes = env.config.fcnet_hidden_sizes
        trunk_fn = nn_util.MLP(fcnet_hidden_sizes, output_layer_activation=True)
        head_mean_fn = nn.Dense(dummy_proposal_output.shape[0])
        head_log_var_fn = nn.Dense(dummy_proposal_output.shape[0])

    else:
        raise NotImplementedError()

    # Configure the proposal structure.
    if env.config.proposal_type == 'VRNN_FILTERING':
        assert env.config.use_bootstrap_initial_distribution, "Error: must use bootstrap/model initialization in VRNN."

        proposal_cls = VrnnFilteringProposal
        n_props = 1  # NOTE - we always use the bootstrap so we just use a single proposal.
        proposal_window_length = 1
        dummy_q_state = None
        dummy_q_inputs = ()

    elif env.config.proposal_type == 'VRNN_SMOOTHING':
        assert env.config.use_bootstrap_initial_distribution, "Error: must use bootstrap/model initialization in VRNN."

        proposal_cls = VrnnSmoothingProposal
        n_props = 1  # NOTE - we always use the bootstrap so we just use a single proposal.
        proposal_window_length = None  # This will use just the current state of and RNN.
        dummy_q_state = None

        dummy_q_inputs = data_encoder.dummy_exposed_state

    else:
        raise NotImplementedError()

    # Not define the input given the structure of the proposal.
    stock_proposal_input = (train_dataset[0], model, dummy_particles, 0, dummy_p_dist, dummy_q_state, dummy_q_inputs)

    # Define the proposal itself.
    print('Defining {} proposals.'.format(n_props))
    proposal = proposal_cls(n_proposals=n_props,
                            stock_proposal_input=stock_proposal_input,
                            dummy_output=dummy_proposal_output,
                            trunk_fn=trunk_fn,
                            head_mean_fn=head_mean_fn,
                            head_log_var_fn=head_log_var_fn,
                            proposal_window_length=proposal_window_length)

    # Initialize the proposal network.
    proposal_params = proposal.init(subkey1)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, env)
    return proposal, proposal_params, rebuild_prop_fn


def define_true_model_and_data(key, env):
    """

    Args:
        key:

    Returns:

    """

    # If using real data, this will be ignored.
    num_trials = env.config.num_trials

    if env.config.synthetic_data:
        emissions_dim = env.config.emissions_dim
        train_dataset_means = 0.5 * np.ones((emissions_dim,))
        output_type = 'GAUSSIAN'
        T = env.config.T  # NOTE - This is the number of transitions in the model (index-0).  There are T+1 variables.

        # To be overwritten later.  just shut the linter up.
        dataset, train_dataset, train_dataset_masks, valid_dataset, valid_dataset_masks = None, None, None, None, None

    else:

        if env.config.dataset in ['piano-midi.pkl', 'nottingham.pkl', 'musedata.pkl', 'jsb.pkl']:
            train_dataset, train_dataset_masks, train_true_states, train_dataset_means = fivo_util.load_piano_data(env.config.dataset, phase='train')
            valid_dataset, valid_dataset_masks, valid_true_states, _ = fivo_util.load_piano_data(env.config.dataset, phase='valid')
            emissions_dim = train_dataset.shape[-1]
            output_type = 'BERNOULLI'
        else:
            raise NotImplementedError()

    # Build up the VRNN.
    RNN_CELL_TYPE = nn.LSTMCell

    latent_dim = env.config.latent_dim
    emissions_encoded_dim = env.config.obs_enc_dim if env.config.obs_enc_dim is not None else latent_dim
    latent_encoded_dim = env.config.latent_enc_dim if env.config.latent_enc_dim is not None else latent_dim
    rnn_state_dim = env.config.rnn_state_dim if env.config.rnn_state_dim is not None else latent_dim
    fcnet_hidden_sizes = env.config.fcnet_hidden_sizes

    # We have to do something a bit funny to the full decoder to make sure that it is approximately correct.
    kernel_init = lambda *args: nn.initializers.xavier_normal()(*args)  # * 0.001

    # Define some dummy place holders.
    val_obs = np.zeros(emissions_dim)
    val_latent = np.zeros(latent_dim)
    val_latent_decoded = np.zeros(latent_encoded_dim)
    val_data_encoded = np.zeros(emissions_encoded_dim)

    # Build up each of the functions.
    prior = nn_util.MLP(fcnet_hidden_sizes + [2 * latent_dim], kernel_init=kernel_init)
    latent_decoder = nn_util.MLP(fcnet_hidden_sizes + [latent_encoded_dim], kernel_init=kernel_init)
    data_encoder = nn_util.MLP(fcnet_hidden_sizes + [emissions_encoded_dim], kernel_init=kernel_init)

    if output_type == 'BERNOULLI':

        clipped_train_dataset_means = np.clip(train_dataset_means, 0.0001, 0.9999)
        clipped_log_odds = np.log(clipped_train_dataset_means) - np.log(1 - clipped_train_dataset_means)
        output_bias_init = lambda *_args: clipped_log_odds

        full_decoder = nn_util.MLP(fcnet_hidden_sizes + [emissions_dim], kernel_init=kernel_init, bias_init=output_bias_init)

    elif output_type == 'GAUSSIAN':
        full_decoder = nn_util.MLP(fcnet_hidden_sizes + [2 * emissions_dim], kernel_init=kernel_init)
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

        valid_true_states = true_states[:env.config.num_val_datasets]
        valid_dataset = dataset[:env.config.num_val_datasets]
        valid_dataset_masks = dataset_masks[:env.config.num_val_datasets]

        train_true_states = true_states[env.config.num_val_datasets]
        train_dataset = dataset[env.config.num_val_datasets:]
        train_dataset_masks = dataset_masks[env.config.num_val_datasets:]

    else:
        train_true_states = None

    return true_model, train_true_states, train_dataset, train_dataset_masks, valid_dataset, valid_dataset_masks


def define_test_model(key, true_model, env):
    """
    This is a bit of a weird function, because we will only ever be using this on real data (in all likelihood).

    Therefore the "true" model is already the model we want.  So this function really just defines the getters and setters for that true model.

    I.e. there is no real "mutation" that happens in this function.

    Args:
        key:
        true_model:
        env:

    Returns:

    """
    # Copy the true model as this is the initial model as well.
    default_model = dc(true_model)

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, env.config.free_parameters)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def do_plot(_param_hist, _loss_hist, _true_loss_em, _true_loss_smc, _true_params, param_figs):
    """
    Not plotting here.
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
    _str = 'Step: {:> 7d},  True Neg-LML: {:> 8.3f},  Pred Neg-LML: {:> 8.3f},  Pred neg FIVO bound {:> 8.3f}'. \
        format(_step, true_lml, pred_lml, pred_fivo_bound)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {:> 8.3f}'.format(em_log_marginal_likelihood)
    print(_str)

    if opt[0] is not None:
        if len(opt[0].target) > 0:
            print('\tModel:  Good luck printing that shite.')
    print()


def get_true_target_marginal(model, data):
    return None


if __name__ == '__main__':

    _phase = 'train'

    for _s in ['piano-midi.pkl', 'nottingham.pkl', 'musedata.pkl', 'jsb.pkl']:
        _dataset, _masks, _true_states, _means = fivo_util.load_piano_data(_s, _phase)

        print('Dataset dimensions (N x T x D): ', _dataset.shape)

        plt.figure()
        plt.imshow(_dataset[0].T)
        plt.title(_s)
        plt.xlim(0, 500)
        plt.pause(0.1)

    print('Done')
