import jax
import argparse
import pickle
import tensorflow as tf
import flax.linen as nn
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random as jr
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd
from flax import optim

import ssm.nn_util as nn_util
from ssm.vrnn.models import VRNN


def get_config():
    """

    Returns:

    """

    # Set up the experiment.
    parser = argparse.ArgumentParser()

    parser.add_argument('--free-parameters', default='params_rnn,params_prior,params_decoder_latent,params_decoder_full,params_encoder_data', type=str)  # CSV.

    parser.add_argument('--proposal-structure', default='VRNN_FILTERING_RESQ', type=str)  # {None/'NONE'/'BOOTSTRAP', 'VRNN_FILTERING_RESQ', }
    parser.add_argument('--proposal-type', default='VRNN_FILTERING', type=str)  # {'VRNN_FILTERING', }

    parser.add_argument('--lr-p', default=0.01, type=float, help="Learning rate of model parameters.")
    parser.add_argument('--lr-q', default=0.0001, type=float, help="Learning rate of proposal parameters.")

    # VRNN architecture args.
    parser.add_argument('--latent-dim', default=10, type=int, help="Dimension of z latent variable.")
    parser.add_argument('--emissions-dim', default=1, type=int, help="Dimension of observed value (Overwritten for real data)")
    parser.add_argument('--latent-enc-dim', default=None, type=int, help="Dimension of encoded latent z variable. (None -> latent-dim)")
    parser.add_argument('--obs-enc-dim', default=None, type=int, help="Dimension of encoded observations. (None -> latent-dim)")
    parser.add_argument('--rnn-state-dim', default=None, type=int, help="Dimension of the deterministic RNN. (None -> latent-dim)")
    parser.add_argument('--fcnet-hidden-sizes', default=None, type=str,
                        help="Layer widths of MLPs. CSV of widths, i.e. '10,10'. (None -> [latent-dim]). ")

    parser.add_argument('--output-type', default='BERNOULLI', type=str, help="Emission distribution family.  {'BERNOULLI', 'GAUSSIAN'}. ")
    parser.add_argument('--rnn-cell-type', default='LSTM', type=str, help="RNN cell family.  {'LSTM', }. ")

    parser.add_argument('--seed', default=10, type=int, help="Seed for initialization.")

    config = parser.parse_args().__dict__

    # Quickly do some type conversion here.

    # If there is no X state dim specified, then just copy the latent dim.
    if config['latent_enc_dim'] is None:
        config['latent_enc_dim'] = config['latent_dim']
    if config['obs_enc_dim'] is None:
        config['obs_enc_dim'] = config['latent_dim']
    if config['rnn_state_dim'] is None:
        config['rnn_state_dim'] = config['latent_dim']

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

    return config


def build_vrnn(env, key, dataset=None):

    # If we have supplied some data, pull out the key statistics.
    if dataset is not None:

        # Quick and dirty test for output type.
        if np.unique(dataset) == (0, 1):
            if env.config.output_type != 'BERNOULLI':
                print('[WARNING]: Defaulting emission/output type to BERNOULLI.')
            output_type = 'BERNOULLI'
        else:
            if env.config.output_type != 'GAUSSIAN':
                print('[WARNING]: Defaulting emission/output type to GAUSSIAN.')
            output_type = 'GAUSSIAN'

        # Grab the emissions shape.
        emissions_dim = dataset.shape[-1]

        # Grab the means of the dataset to define the odds.
        assert len(dataset.shape) == 3, "Dataset must be (N x T x D)."
        dataset_means = np.mean(np.mean(dataset, axis=0), axis=0)

    else:
        output_type = env.config.output_type
        emissions_dim = env.config.emissions_dim
        dataset_means = np.zeros(emissions_dim)

    # Grab the RNN type.
    if env.config.rnn_cell_type == 'LSTM':
        rnn_cell_type = nn.LSTMCell
    else:
        raise NotImplementedError()

    # Pull out the dimensions.
    latent_dim = env.config.latent_dim
    emissions_encoded_dim = env.config.obs_enc_dim
    latent_encoded_dim = env.config.latent_enc_dim
    rnn_state_dim = env.config.rnn_state_dim
    fcnet_hidden_sizes = env.config.fcnet_hidden_sizes

    # Use the standard initializer.
    kernel_init = lambda *args: nn.initializers.xavier_normal()(*args)

    # generate the keys for initialization.
    key, *subkeys = jr.split(key, num=7)

    # Define some dummy place holders.
    val_obs = np.zeros(emissions_dim)
    val_latent = np.zeros(latent_dim)
    val_latent_decoded = np.zeros(latent_encoded_dim)
    val_data_encoded = np.zeros(emissions_encoded_dim)

    # Build up each of the functions, apart from the RNN.
    prior = nn_util.MLP(fcnet_hidden_sizes + [2 * latent_dim], kernel_init=kernel_init)
    latent_decoder = nn_util.MLP(fcnet_hidden_sizes + [latent_encoded_dim], kernel_init=kernel_init)
    data_encoder = nn_util.MLP(fcnet_hidden_sizes + [emissions_encoded_dim], kernel_init=kernel_init)

    if output_type == 'BERNOULLI':
        clipped_train_dataset_means = np.clip(dataset_means, 0.0001, 0.9999)
        clipped_log_odds = np.log(clipped_train_dataset_means) - np.log(1 - clipped_train_dataset_means)
        output_bias_init = lambda *_args: clipped_log_odds
        full_decoder = nn_util.MLP(fcnet_hidden_sizes + [emissions_dim], kernel_init=kernel_init, bias_init=output_bias_init)

    elif output_type == 'GAUSSIAN':
        full_decoder = nn_util.MLP(fcnet_hidden_sizes + [2 * emissions_dim], kernel_init=kernel_init)
    else:
        raise NotImplementedError()

    # Building up the RNN requires a little more of a delicate touch.
    rnn = rnn_cell_type()
    rnn_carry = rnn.initialize_carry(subkeys[0], batch_dims=(), size=rnn_state_dim)
    input_rnn = tuple((rnn_carry, np.concatenate((val_latent_decoded, val_data_encoded))))

    # Work out the dimensions of the inputs to each function.
    input_latent_decoder = val_latent
    input_data_encoder = val_obs
    input_full_decoder = np.concatenate((rnn_carry[1], val_latent_decoded))  # TODO - need to parse the carry for the case of the GRU.
    input_prior = rnn_carry[1]

    # Initialize the functions and grab the parameters.
    params_rnn = rnn.init(subkeys[1], input_rnn[0], input_rnn[1])
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
    model = VRNN(num_latent_dims=latent_dim,
                 num_emission_dims=emissions_dim,
                 latent_enc_dim=latent_encoded_dim,
                 obs_enc_dim=emissions_encoded_dim,
                 rnn_state_dim=rnn_state_dim,

                 output_type=output_type,

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

    return model
