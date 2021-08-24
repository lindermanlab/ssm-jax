import jax.numpy as np
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp

from ssm.models.base import HMM, _make_standard_hmm


def make_poisson_hmm(num_states,
                     emission_dim,
                     initial_state_probs=None,
                     initial_state_logits=None,
                     transition_matrix=None,
                     transition_logits=None,
                     emission_log_rates=None):
    """
    Helper function to create a Gaussian HMM
    """

    # Initialize the basics
    initial_dist, transition_dist = \
        _make_standard_hmm(num_states,
                           initial_state_probs=initial_state_probs,
                           initial_state_logits=initial_state_logits,
                           transition_matrix=transition_matrix,
                           transition_logits=transition_logits)

    # Initialize the Gaussian emissions
    if emission_log_rates is None:
        emission_log_rates = np.zeros((num_states, emission_dim))

    emissions_dist = tfp.distributions.Independent(
        tfp.distributions.Poisson(log_rate=emission_log_rates),
        reinterpreted_batch_ndims=1,
    )

    return HMM(num_states, initial_dist, transition_dist, emissions_dist)


def initialize_poisson_hmm(rng, num_states, data, **kwargs):
    """
    Initializes a Gaussian in a semi-data-intelligent manner.
    """

    # Pick random data points as the means
    num_timesteps, emission_dim = data.shape
    assignments = jr.choice(rng, num_states, shape=(num_timesteps,))
    rates = np.row_stack(
        [data[assignments == k].mean(axis=0) for k in range(num_states)]
    )

    return make_poisson_hmm(
        num_states, emission_dim,
        emission_log_rates=np.log(rates),
        **kwargs)
