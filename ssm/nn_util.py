"""
Module containing utility scripts for neural networks.
"""

import jax
import jax.numpy as np
import flax.linen as nn
from tensorflow_probability.substrates.jax import distributions as tfd

from typing import (NamedTuple, Any, Callable, Sequence, Iterable, List, Optional, Tuple,
                    Set, Type, Union, TypeVar, Generic, Dict)

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any



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
    Define a simple fully connected MLP with ReLU activations.
    """
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.glorot_normal
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init, )(x))
        x = nn.Dense(self.features[-1])(x)
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
