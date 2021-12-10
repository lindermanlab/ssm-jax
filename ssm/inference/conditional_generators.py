"""
Learnable parameter generators for use as proposal distributions.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import ssm.nn_util as nn_util
from jax import flatten_util

# Specific imports for here.
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from ssm.utils import Verbosity

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def build_independent_gaussian_generator(dummy_input, dummy_output,
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

    input_flat, _ = flatten_util.ravel_pytree(dummy_input)
    input_flat_shape = input_flat.shape
    input_flat_dim = input_flat_shape[0]

    output_flat, unravel_output = flatten_util.ravel_pytree(dummy_output)
    output_flat_shape = output_flat.shape
    output_flat_dim = output_flat_shape[0]

    # If no trunk if defined, then use the identity.
    if trunk_fn is None:
        trunk_fn = nn_util.Identity(input_flat_dim)

    # If no head mean function is specified, default to an affine transformation.
    if head_mean_fn is None:
        head_mean_fn = nn.Dense(output_flat_dim)

    # If no head mean function is specified, default to an affine transformation.
    if head_log_var_fn is None:
        head_log_var_fn = nn.Dense(output_flat_dim)

    # # This MVN can only handle vector events, so we need to type check for that.
    # assert len(output_dim) == 1, "Output dimensions must be one."

    # Type check to make sure that the mean function produces the right size.
    _dim = head_mean_fn.features[-1] if hasattr(head_mean_fn.features, '__iter__') else head_mean_fn.features
    assert output_flat_dim == _dim, \
        'Error: head mean output dimensions not equal to the target output dimensions.'

    # Type check to make sure that the variance function produces the right size.
    _dim = head_log_var_fn.features[-1] if hasattr(head_log_var_fn.features, '__iter__') else head_log_var_fn.features
    assert output_flat_dim == _dim, \
        'Error: head variance output dimensions not equal to the target output dimensions.'

    class IndependentGaussianGenerator(nn.Module):

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
            self.unravel_output = unravel_output

        def __call__(self, inputs):
            """
            Equivalent to the `.forward` method in PyTorch.  Generates the parameters of an independent
            multivariate Gaussian distribution as a function of the inputs.

            Args:
                inputs:

            Returns:

            """
            mean, var = self._generate_distribution_parameters(inputs)
            dist = jax.tree_map(lambda _mu, _var: tfd.MultivariateNormalDiag(loc=_mu, scale_diag=np.sqrt(_var)),
                                mean, var)
            return dist

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
            is_batched = (flatten_util.ravel_pytree(inputs[0])[0].shape[0] == input_flat_dim)

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
            inputs_flat = nn_util.vectorize_pytree(inputs)

            # Apply the trunk.
            trunk_output = self.trunk_fn(inputs_flat)

            # Get the mean.
            mean_output_flat = self.head_mean_fn(trunk_output)
            mean_output = mean_output_flat

            # Get the variance output and reshape it.
            var_output_flat = self.head_log_var_fn(trunk_output)
            var_output = var_output_flat  # TODO - removed exp from output.

            # Unravel the output to the original shape.
            mean_output_shaped = unravel_output(mean_output)
            var_output_shaped = unravel_output(var_output)

            return mean_output_shaped, var_output_shaped

    return IndependentGaussianGenerator()

