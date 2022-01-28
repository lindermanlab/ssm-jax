"""
Module containing utility scripts for neural networks.
"""

import jax
import jax.numpy as np
import flax.linen as nn
from tensorflow_probability.substrates.jax import distributions as tfd

from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)

from typing import (Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)
from flax.core.scope import CollectionFilter, DenyList, Variable, VariableDict, FrozenVariableDict, union_filters
from flax.linen.initializers import orthogonal, zeros

import jax


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any
RNGSequences = Dict[str, PRNGKey]


def vectorize_pytree(*args):
    """
    Flatten an arbitrary PyTree into a vector.
    :param args:
    :return:
    """
    flat_tree, _ = jax.tree_util.tree_flatten(args)
    flat_vs = [x.flatten() for x in flat_tree]
    return np.concatenate(flat_vs, axis=0)


class MLP(nn.Module):
    """
    Define a simple fully connected MLP (default ReLU activations).
    """
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.glorot_normal
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    output_layer_activation: bool = False
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)

        if self.output_layer_activation:
            x = self.activation(x)

        return x


class Identity(nn.Module):
    """
    A layer which passes the input through unchanged.
    """
    features: int

    def __call__(self, inputs):
        return inputs


class Static(nn.Module):
    """
    A layer which just returns some static parameters.
    """
    features: int
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('bias',
                            self.bias_init,
                            (self.features, ))
        return kernel


class RnnWithReadoutLayer(nn.Module):
    """
    Combine an RNN with a readout layer to encode the (exposed) hidden state.
    """
    emissions_dim: int
    latent_dim: int = 64
    _rnn_cell: nn.recurrent.RNNCellBase = nn.LSTMCell
    _readout_network: nn.Module = nn.Dense

    def setup(self):
        self.rnn_cell = self._rnn_cell()
        self.readout_network = self._readout_network(self.emissions_dim)

    def initialize_carry(self, rng, batch_dims=(), init_fn=zeros):
        """

        Args:
            rng:
            batch_dims:
            size:
            init_fn:

        Returns:

        """
        return self._rnn_cell().initialize_carry(rng, batch_dims, self.latent_dim, init_fn)

    def __call__(self, carry, x):
        """

        Args:
            carry:
            x:

        Returns:

        """
        rnn_carry, rnn_carry_exposed = self.rnn_cell(carry, x)
        rnn_y_out = self.readout_network(rnn_carry_exposed)
        return rnn_carry, rnn_y_out


# if __name__ == '__main__':
#
#     key = jax.random.PRNGKey(0)
#     _emissions_dim = 4
#     wrapped = RnnWithReadoutLayer(_emissions_dim)
#     init_carry = wrapped.initialize_carry(key)
#     params1 = wrapped.init(key, init_carry, np.zeros(_emissions_dim))
#     p = 0

