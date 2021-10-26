"""
Base Component Classes
======================

Just a placeholder while we refactor LDS
"""

import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, tree_util, vmap
from jax.tree_util import register_pytree_node_class
from ssm.distributions.expfam import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


@register_pytree_node_class
class DiscreteComponent:
    def __init__(self, distribution, num_states):
        assert isinstance(distribution, tfp.distributions.Distribution)
        self.distribution = distribution
        self.num_states = num_states

    def exact_m_step(self, data, posterior, prior=None):
        """Return a new distribution after applying the M step in EM.
        """
        return NotImplementedError

    def permute(self, permutation):
        """Permute the discrete states of the underlying distribution.
        """
        return NotImplementedError

    def initialize(self, data):
        """Initialize the distribution in a data-aware manner.
        """
        return NotImplementedError

    def tree_flatten(self):
        children = (self.distribution,)
        aux_data = (self.num_states,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, = aux_data
        distribution, = children
        return cls(distribution=distribution,
                   num_states=num_states)


@register_pytree_node_class
class ContinuousComponent:
    def __init__(self, distribution):
        assert isinstance(distribution, tfp.distributions.Distribution)
        self.distribution = distribution

    def exact_m_step(self, data, posterior, prior=None):
        """Return a new distribution after applying the M step in EM.
        """
        raise NotImplementedError

    def initialize(self, data):
        """Initialize the distribution in a data-aware manner.
        """
        return NotImplementedError

    def tree_flatten(self):
        children = (self.distribution,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, = children
        return cls(distribution=distribution)
