"""
Base model class.
"""

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


class SSM(object):
    def initial_dist(self):
        raise NotImplementedError

    def dynamics_dist(self, state):
        raise NotImplementedError

    def emissions_dist(self, state):
        raise NotImplementedError

    def log_probability(self, states, data):
        """
        Compute the log joint probability of a set of states and data
        """
        lp = 0
        lp += self.initial_dist().log_prob(states[0])
        lp += self.emissions_dist(states[0]).log_prob(data[0])

        def _step(carry, args):
            prev_state, lp = carry
            state, emission = args
            lp += self.dynamics_dist(prev_state).log_prob(state)
            lp += self.emissions_dist(state).log_prob(emission)
            return (state, lp), None

        (_, lp), _ = lax.scan(_step, (states[0], lp), zip(states[1:], data[1:]))

    def sample(self, key, num_steps, covariates=None, initial_state=None):
        """
        Sample the state space model for specified number of steps.
        """
        if initial_state is None:
            key1, key = jr.split(key, 2)
            initial_state = self.initial_dist().sample(seed=key1)

        def _step(state, key):
            key1, key2 = jr.split(key, 2)
            emission = self.emissions_dist(state).sample(seed=key1)
            next_state = self.dynamics_dist(state).sample(seed=key2)
            return next_state, (state, emission)

        keys = jr.split(key, num_steps)
        _, (states, emissions) = lax.scan(_step, initial_state, keys)

        return states, emissions
