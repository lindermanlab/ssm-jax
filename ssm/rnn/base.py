from jax.tree_util import register_pytree_node_class
import jax.random as jr
import jax.numpy as np
import flax.linen as nn

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from ssm.base import SSM


@register_pytree_node_class
class DeterministicRNN(SSM):
    def __init__(self, rnn_params, initial_state):
        self._rnn_params = rnn_params
        self._initial_state = initial_state

    def tree_flatten(self):
        children = (self._rnn_params, self._initial_state)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        rnn_params, init_state = children
        return cls(rnn_params=rnn_params, initial_state=init_state)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # We have to be a little fancy since this classmethod
        # is inherited by subclasses with different constructors.
        obj = object.__new__(cls)
        rnn_params, initial_state = children  # TODO: this won't scale well
        DeterministicRNN.__init__(obj, rnn_params, initial_state)
        return obj

    def initial_distribution(self):
        return tfd.Deterministic(self._initial_state)

    def dynamics_distribution(self, state, covariates):
        return NotImplementedError

    def emissions_distribution(self, state):
        return tfd.Deterministic(state)
