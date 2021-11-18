"""
FIVO implementation for join state-space inference and parameter learning in SSMs.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import ssm.nn_util as nn_util

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

    :param dummy_input:
    :param dummy_output:
    :param trunk_fn:
    :param head_mean_fn:
    :param head_log_var_fn:
    :return:
    """

    input_dim = dummy_input.shape
    output_dim = dummy_output.shape

    input_dim_flat = nn_util.vectorize_pytree(dummy_input).shape[0]
    output_dim_flat = output_dim[0]

    # If no trunk if defined, then use the identity.
    if trunk_fn is None:
        trunk_fn = nn_util.Identity(input_dim_flat)

    # If no head mean function is specified, default to an affine transformation.
    if head_mean_fn is None:
        head_mean_fn = nn.Dense(output_dim_flat)

    # If no head mean function is specified, default to an affine transformation.
    if head_log_var_fn is None:
        head_log_var_fn = nn.Dense(output_dim_flat)

    # This MVN can only handle vector events, so we need to type check for that.
    assert len(output_dim) == 1, "Output dimensions must be one."

    # Type check to make sure that the mean function produces the right size.
    _dim = head_mean_fn.features[-1] if hasattr(head_mean_fn.features, '__iter__') else head_mean_fn.features
    assert output_dim_flat == _dim, \
        'Error: head mean output dimensions not equal to the target output dimensions.'

    # Type check to make sure that the variance function produces the right size.
    _dim = head_log_var_fn.features[-1] if hasattr(head_log_var_fn.features, '__iter__') else head_log_var_fn.features
    assert output_dim_flat == _dim, \
        'Error: head variance output dimensions not equal to the target output dimensions.'

    class IndependentGaussianGenerator(nn.Module):

        def setup(self):

            # Inscribe this stuff.
            self.trunk_fn = trunk_fn
            self.head_mean_fn = head_mean_fn
            self.head_log_var_fn = head_log_var_fn

        def __call__(self, inputs):
            mean, var = self._generate_distribution_parameters(inputs)
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=np.sqrt(var))
            return dist

        def _generate_distribution_parameters(self, inputs):

            # If the shape is equal to the input dimensions then there is no batch dimension
            # and we can call the forward function as is.  Otherwise we need to do a vmap
            # over the batch dimension.
            is_batched = (nn_util.vectorize_pytree(inputs[0]).shape[0] == input_dim_flat)

            if is_batched:
                return vmap(self._call_single)(inputs)
            else:
                return self._call_single(inputs)

        def _call_single(self, inputs):

            # Flatten the input.
            inputs_flat = nn_util.vectorize_pytree(inputs)

            # Apply the trunk.
            trunk_output = self.trunk_fn(inputs_flat)

            # Get the mean.
            mean_output_flat = self.head_mean_fn(trunk_output)
            mean_output = mean_output_flat

            # Get the variance output and reshape it.
            var_output_flat = self.head_log_var_fn(trunk_output)
            var_output = np.exp(var_output_flat)

            return mean_output, var_output

    return IndependentGaussianGenerator()


def _test():
    dummy_input = np.ones([12, 13])
    dummy_output = np.ones([14, 15]).flatten()

    trunk_fn = None  # MLP(features=(3, 4, 5))
    mean_fn = nn_util.MLP(features=(6, 7, nn_util.vectorize_pytree(dummy_output).shape[0]))
    var_fn = nn_util.MLP(features=(9, 10, nn_util.vectorize_pytree(dummy_output).shape[0]))

    model = build_independent_gaussian_generator(dummy_input=dummy_input, dummy_output=dummy_output,
                                                 trunk_fn=trunk_fn, head_mean_fn=mean_fn, head_log_var_fn=var_fn)

    key1, key2 = jr.split(jr.PRNGKey(0), 2)
    x_single = jr.uniform(key1, dummy_input.shape)
    params = model.init(key2, x_single)

    # Test for a single point.
    y_single = model.apply(params, x_single)
    assert y_single.loc.shape == dummy_output.shape

    # Test for batched output.
    n_points = 16
    x_batch = jr.uniform(key1, (n_points, *dummy_input.shape))
    y_batch = model.apply(params, x_batch)
    assert y_batch.loc.shape == (n_points, *dummy_output.shape)

    print('initialized parameter shapes:\n', jax.tree_map(np.shape, params))
    print('output:\n', y_single)

    import timeit

    st = timeit.default_timer()
    for _ in range(100):
        key1, subkey = jr.split(key1)
        x_single = x_single + jr.normal(key1, shape=x_single.shape)
        model.apply(params, x_single)
    print('Non-jit time: ', timeit.default_timer() - st)

    apply = jax.jit(model.apply)
    st = timeit.default_timer()
    for _ in range(100):
        key1, subkey = jr.split(key1)
        x_single = x_single + jr.normal(key1, shape=x_single.shape)
        apply(params, x_single)
    print('Jit time:     ', timeit.default_timer() - st)

    p = 0


if __name__ == '__main__':
    _test()
