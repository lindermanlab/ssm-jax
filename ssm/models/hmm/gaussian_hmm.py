import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import lax, value_and_grad, jit, vmap, grad
from jax.tree_util import register_pytree_node, register_pytree_node_class
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates import jax as tfp

from functools import partial, wraps
from textwrap import dedent
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm.auto import trange
from enum import IntEnum

from .base import _StandardHMM


@register_pytree_node_class
class GaussianHMM(_StandardHMM):
    def __init__(
        self,
        num_states,
        emission_dim,
        initial_state_probs=None,
        initial_state_logits=None,
        transition_matrix=None,
        transition_logits=None,
        emission_means=None,
        emission_covariances=None,
        emission_scale_trils=None,
    ):
        """ """
        _StandardHMM.__init__(
            self,
            num_states,
            initial_state_probs=initial_state_probs,
            initial_state_logits=initial_state_logits,
            transition_matrix=transition_matrix,
            transition_logits=transition_logits,
        )

        # Initialize the Gaussian emissions
        self.emission_dim = emission_dim

        if emission_means is None:
            emission_means = np.zeros((num_states, emission_dim))

        if emission_scale_trils is None:
            if emission_covariances is None:
                emission_scale_trils = np.tile(np.eye(emission_dim), (num_states, 1, 1))
            else:
                emission_scale_trils = np.linalg.cholesky(emission_covariances)

        self._emissions_dist = tfp.distributions.MultivariateNormalTriL(
            loc=emission_means, scale_tril=emission_scale_trils
        )

    def emissions_dist(self, state):
        return self._emissions_dist[state]

    @property
    def emission_means(self):
        return self._emissions_dist.mean()

    @property
    def emission_covariances(self):
        return self._emissions_dist.covariance()

    @classmethod
    def initialize(cls, rng, num_states, data):
        # Pick random data points as the means
        num_timesteps, emission_dim = data.shape
        inds = jr.choice(rng, num_timesteps, shape=(num_states,), replace=False)
        means = data[inds]

        # Set the covariance to a fraction of the marginal covariance
        cov = np.cov(data, rowvar=False)
        scale_tril = np.tile(np.linalg.cholesky(cov) / num_states, (num_states, 1, 1))

        return cls(
            num_states,
            emission_dim,
            emission_means=means,
            emission_scale_trils=scale_tril,
        )

    # Note: using tree_flatten and tree_unflatten for two purposes:
    # 1. Make GaussianHMM a PyTree (and hence jittable)
    # 2. Get the unconstrained parameters of the model
    def tree_flatten(self):
        children = (
            self._initial_dist.logits_parameter(),
            self._dynamics_dist.logits_parameter(),
            self._emissions_dist.loc,
            self._emissions_dist.scale_tril,
        )
        aux_data = self.num_states, self.emission_dim
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, emission_dim = aux_data
        (
            initial_state_logits,
            transition_logits,
            emission_means,
            emission_scale_trils,
        ) = children

        # Construct a new HMM
        return cls(
            num_states,
            emission_dim,
            initial_state_logits=initial_state_logits,
            transition_logits=transition_logits,
            emission_means=emission_means,
            emission_scale_trils=emission_scale_trils,
        )
