from collections import namedtuple
from enum import IntEnum
from functools import partial, wraps
from textwrap import dedent

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
import matplotlib.pyplot as plt
from jax import grad, jit, lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node, register_pytree_node_class
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

from .base import HMM, _make_standard_hmm


def make_gaussian_hmm(num_states,
                      emission_dim,
                      initial_state_probs=None,
                      initial_state_logits=None,
                      transition_matrix=None,
                      transition_logits=None,
                      emission_means=None,
                      emission_covariances=None,
                      emission_scale_trils=None):
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
    if emission_means is None:
        emission_means = np.zeros((num_states, emission_dim))

    if emission_scale_trils is None:
        if emission_covariances is None:
            emission_scale_trils = np.tile(np.eye(emission_dim), (num_states, 1, 1))
        else:
            emission_scale_trils = np.linalg.cholesky(emission_covariances)

    emission_dist = \
    tfp.distributions.MultivariateNormalTriL(loc=emission_means,
                                             scale_tril=emission_scale_trils)

    return HMM(num_states, initial_dist, transition_dist, emission_dist)


def initialize_gaussian_hmm(rng, num_states, data, **kwargs):
    """
    Initializes a Gaussian in a semi-data-intelligent manner.
    """
    
    # Pick random data points as the means
    num_timesteps, emission_dim = data.shape
    inds = jr.choice(rng, num_timesteps, shape=(num_states,), replace=False)
    means = data[inds]

    # from sklearn.cluster import KMeans
    # km = KMeans(num_states)
    # km.fit(data)
    # means = km.cluster_centers_

    # Set the covariance to a fraction of the marginal covariance
    cov = np.cov(data, rowvar=False)
    scale_tril = np.tile(np.linalg.cholesky(cov) / num_states, (num_states, 1, 1))

    return make_gaussian_hmm(
        num_states, emission_dim,
        emission_means=means,
        emission_scale_trils=scale_tril,
        **kwargs)