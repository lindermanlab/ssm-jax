"""
FIVO implementation for join state-space inference and parameter learning in SSMs.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

# Specific imports for here.
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from ssm.utils import Verbosity
from typing import NamedTuple, Any

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


# Linen stuff.
from tensorflow_probability.substrates.jax import distributions as tfd
import ssm.snax.snax as snax
from ssm.snax.snax.nn import MLP, Affine, Identity


class GaussianGeneratorParams(NamedTuple):
    trunk_fn: Any
    head_mean_fn: Any
    head_log_var_fn: Any


def IndependentGaussianGenerator(dummy_input, dummy_output,
                                 trunk_fn=None, head_mean_fn=None, head_log_var_fn=None):

    input_dim = snax.vectorize_pytree(dummy_input).shape[0]
    output_dim = snax.vectorize_pytree(dummy_output).shape[0]

    # If no trunk if defined, then use the identity.
    if trunk_fn is None:
        trunk_fn = snax.nn.Identity()

    # If no head mean function is specified, default to an affine transformation.
    if head_mean_fn is None:
        head_mean_fn = snax.nn.Affine(output_dim)

    # If no head mean function is specified, default to an affine transformation.
    if head_log_var_fn is None:
        head_log_var_fn = snax.nn.Affine(output_dim)

    # Type check to make sure that the mean function produces the right size.
    assert output_dim == head_mean_fn.get_output_dims(trunk_fn.get_output_dims(input_dim)), \
        'Error: head mean output dimensions not equal to the target output dimensions.'

    # Type check to make sure that the variance function produces the right size.
    assert output_dim == head_log_var_fn.get_output_dims(trunk_fn.get_output_dims(input_dim)), \
        'Error: head variance output dimensions not equal to the target output dimensions.'

    def init(key):
        # Split the key.
        k1, k2, k3 = jr.split(key, num=3)

        # Initialize the trunk.
        trunk_output_size, trunk_fn_params = trunk_fn.init(k1, input_dim)

        # Initialize the two head networks.
        _, head_mean_fn_params = head_mean_fn.init(k2, trunk_output_size)
        _, head_log_var_fn_params = head_log_var_fn.init(k3, trunk_output_size)

        return GaussianGeneratorParams(trunk_fn_params, head_mean_fn_params, head_log_var_fn_params)

    def _call_single(params, inputs):

        # Flatten the input.
        inputs_flat = np.reshape(inputs, (-1, ))

        # Apply the trunk.
        trunk_output = trunk_fn.apply(params.trunk_fn, inputs_flat)

        # Get the mean.
        mean_output_flat = head_mean_fn.apply(params.head_mean_fn, trunk_output)
        mean_output = mean_output_flat

        # Get the variance output and reshape it.
        var_output_flat = head_log_var_fn.apply(params.head_log_var_fn, trunk_output)
        var_output = np.exp(var_output_flat)

        return mean_output, var_output

    def apply(params, inputs):
        params = generate_distribution_parameters(params, inputs)
        dist = tfd.MultivariateNormalDiag(loc=params[0], scale_diag=np.sqrt(params[1]))
        return dist

    def get_output_dims(input_shape=None):
        return output_dim

    def generate_distribution_parameters(params, inputs):

        # If the shape is equal to the input dimensions then there is no batch dimension
        # and we can call the forward function as is.  Otherwise we need to do a vmap
        # over the batch dimension.
        if inputs.shape == input_dim:
            return _call_single(params, inputs)
        else:
            return vmap(_call_single, in_axes=(None, 0))(params, inputs)

    return snax.Module(init, apply, get_output_dims, generate_distribution_parameters)


def test():
    dummy_input = np.ones([12, 13])
    dummy_output = np.ones([14, 15])

    trunk_fn_in = snax.MLP(layer_dims=[3, 4, 5])
    mean_fn = snax.MLP(layer_dims=[6, 7, snax.vectorize_pytree(dummy_output).shape[0]])
    var_fn = snax.MLP(layer_dims=[9, 10, snax.vectorize_pytree(dummy_output).shape[0]])

    model = IndependentGaussianGenerator(dummy_input=dummy_input, dummy_output=dummy_output,
                                         trunk_fn=trunk_fn_in, head_mean_fn=mean_fn, head_log_var_fn=var_fn)

    key1, key2 = jr.split(jr.PRNGKey(0), 2)
    x = jr.uniform(key1, (32, *dummy_input.shape))
    params = model.init(key2)
    y = model.apply(params, x)

    print('initialized parameter shapes:\n', jax.tree_map(np.shape, params))
    print('output:\n', y)

    import timeit
    st = timeit.default_timer()
    for _ in range(100):
        key1, subkey = jr.split(key1)
        x = x + jr.normal(key1, shape=x.shape)
        model.apply(params, x)
    print(timeit.default_timer() - st)


    p = 0


if __name__ == '__main__':
    test()
