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
class PoissonHMM(_StandardHMM):
    def __init__(
        self,
        num_states,
        emission_dim,
        initial_state_probs=None,
        initial_state_logits=None,
        transition_matrix=None,
        transition_logits=None,
        emission_rates=None,
        emission_log_rates=None,
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

        if emission_log_rates is None:
            if emission_rates is None:
                emission_log_rates = np.zeros((num_states, emission_dim))
            else:
                emission_log_rates = np.log(emission_rates)

        self._emissions_dist = tfp.distributions.Independent(
            tfp.distributions.Poisson(log_rate=emission_log_rates),
            reinterpreted_batch_ndims=1,
        )

    def emissions_dist(self, state):
        return self._emissions_dist[state]

    @property
    def emission_rates(self):
        return self._emissions_dist.mean()

    @classmethod
    def initialize(cls, rng, num_states, data):
        # Pick random data points as the means
        num_timesteps, emission_dim = data.shape
        assignments = jr.choice(rng, num_states, shape=(num_timesteps,))
        rates = np.row_stack(
            [data[assignments == k].mean(axis=0) for k in range(num_states)]
        )

        return cls(num_states, emission_dim, emission_rates=rates)

    # Note: using tree_flatten and tree_unflatten for two purposes:
    # 1. Make GaussianHMM a PyTree (and hence jittable)
    # 2. Get the unconstrained parameters of the model
    def tree_flatten(self):
        children = (
            self._initial_state_dist.logits_parameter(),
            self._dynamics_dist.logits_parameter(),
            self._emissions_dist.distribution.log_rate_parameter(),
        )
        aux_data = self.num_states, self.emission_dim
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, emission_dim = aux_data
        initial_state_logits, transition_logits, emission_log_rates = children

        # Construct a new HMM
        return cls(
            num_states,
            emission_dim,
            initial_state_logits=initial_state_logits,
            transition_logits=transition_logits,
            emission_log_rates=emission_log_rates,
        )

    def tree_unflatten_inplace(self, aux_data, children):
        num_states, emission_dim = aux_data
        initial_state_logits, transition_logits, emission_log_rates = children

        # Construct a new HMM
        self.__init__(
            num_states,
            emission_dim,
            initial_state_logits=initial_state_logits,
            transition_logits=transition_logits,
            emission_log_rates=emission_log_rates,
        )
