# Stolen from Andy's smc branch

import jax
from jax import vmap
import jax.numpy as np
import flax.linen as nn
from jax.tree_util import register_pytree_node_class

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
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.glorot_normal()
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
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.features, ))
        return kernel

def build_gaussian_network(input_dim, output_dim,
                        trunk_fn=None, head_mean_fn=None, head_log_var_fn=None):
    """
    Build a nn.Module that implements a conditional Gaussian density.  This function basically wraps a generator and
    allows for more flexible definition of link functions.
    Args:
        dummy_input (ndarray):                  Correctly shaped ndarray of dummy input values that is the same size as
                                                the input.  No leading batch dimension.
        dummy_output (ndarray):                 Correctly shaped ndarray of dummy output values that is the same size as
                                                the input.  No leading batch dimension.
        trunk_fn (nn.Module or None):           Module that is applied to input to create embedding.  Defaults to I.
        head_mean_fn (nn.Module or None):       Module applied to embedded values to create mean.  Defaults to affine.
        head_log_var_fn (nn.Module or None):    Module applied to embedded values to create variance.  Defaults to
                                                affine.
    Returns:
        (nn.Module):                            Proposal object.
    """

    # input_dim = dummy_input.shape
    # output_dim = dummy_output.shape

    input_dim_flat = input_dim #vectorize_pytree(dummy_input).shape[0]
    output_dim_flat = output_dim #[0]

    # If no trunk if defined, then use the identity.
    if trunk_fn is None:
        trunk_fn = Identity(input_dim_flat)

    # If no head mean function is specified, default to an affine transformation.
    if head_mean_fn is None:
        head_mean_fn = nn.Dense(output_dim_flat)

    # If no head mean function is specified, default to an affine transformation.
    if head_log_var_fn is None:
        head_log_var_fn = nn.Dense(output_dim_flat)

    @register_pytree_node_class
    class PotentialGenerator(nn.Module):

        def setup(self):
            """
            Method required by Flax/Linen to set up the neural network.  Inscribes the layers defined in the outer
            scope.
            Returns:
                - None
            """
            # Inscribe this stuff.
            self.trunk_fn = trunk_fn
            self.head_mean_fn = head_mean_fn
            self.head_log_var_fn = head_log_var_fn

        def tree_flatten(self):
            children = tuple()
                        # (self.trunk_fn,
                        # self.head_mean_fn,
                        # self.head_log_var_fn)
            aux_data = None
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls()

        def __call__(self, inputs):
            """
            Equivalent to the `.forward` method in PyTorch.  Generates the parameters of an independent
            multivariate Gaussian distribution as a function of the inputs.
            Args:
                inputs:
            Returns:
            """
            J_diag, h = self._generate_distribution_parameters(inputs)
            return (J_diag, h)

        def _generate_distribution_parameters(self, inputs):
            """
            Map over the inputs if there are multiple inputs, otherwise, apply it to a single input.
            Whether there is a batch dimension is established by whether the whole shape of the input equals the
            pre-specified input shape.  If they are unequal, assumes that the first dimension is a batch dimension.
            Args:
                inputs (ndarray):   Possibly batched set of inputs (if batched, first dimension is batch dimension).
            Returns:
                (ndarray):          Possibly batched set of Gaussian parameters (if batched, first dimension is batch
                                    dimension).
            """

            # If the shape is equal to the input dimensions then there is no batch dimension
            # and we can call the forward function as is.  Otherwise we need to do a vmap
            # over the batch dimension.
            is_batched = (vectorize_pytree(inputs[0]).shape[0] == input_dim_flat)
            if is_batched:
                return vmap(self._call_single)(inputs)
            else:
                return self._call_single(inputs)

        def _call_single(self, inputs):
            """
            Args:
                inputs (ndarray):   Single input data point (NO batch dimension).
            Returns:
                (ndarray):          Single mean and variance VECTORS representing the parameters of a single
                                    independent multivariate normal distribution.
            """

            # Flatten the input.
            inputs_flat = vectorize_pytree(inputs)

            # Apply the trunk.
            trunk_output = self.trunk_fn(inputs_flat)

            # Get the mean.
            h_flat = self.head_mean_fn(trunk_output)
            h = h_flat

            # Get the variance output and reshape it.
            var_output_flat = self.head_log_var_fn(trunk_output)
            J_diag = np.exp(var_output_flat)

            return (J_diag, h)

    return PotentialGenerator()