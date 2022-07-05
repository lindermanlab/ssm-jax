import jax
from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import flax.linen as nn

import ssm.distributions as ssmd
from ssm.distributions import GaussianLinearRegression, glm
from ssm.lds.emissions import Emissions
from ssm.nn_util import Identity, MLP, vectorize_pytree, softplus

@register_pytree_node_class
class NeuralNetworkEmissions(Emissions):

    def __init__(self,
                 in_dim,
                 out_dim,
                 emissions_network=None,
                 parameters=None) -> None:
        if emissions_network is not None:
            self._emissions_network = emissions_network
        else:
            self._emissions_network = build_gaussian_emissions(in_dim, out_dim)
        self._params = parameters
        self._in_dim = in_dim
        self._out_dim = out_dim

    def tree_flatten(self):
        children = (self._in_dim, 
                    self._out_dim, 
                    self._emissions_network, 
                    self._params)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # emissions_network, params = children
        return cls(*children)

    @property
    def params(self):
        return self._params

    @property
    def latent_dim(self):
        return self._in_dim
    
    @property
    def emissions_shape(self):
        return (self._out_dim,)

    def init(self, key, data):
        return self._emissions_network.init(key, data)        

    def update_params(self, params):
        self._params = params

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of emission y_t
        given state x_t and (optionally) covariates u_t.
        Args:
            state (float): continuous state
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        cov, loc = self._emissions_network.apply(self._params, state)
        return tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)


    def m_step(self,
               data,
               posterior,
               covariates=None,
               metadata=None,
               key=None):
        """
        Does nothing
        """
        return self

def build_gaussian_emissions(input_dim, output_dim,
                        trunk_fn=None, head_mean_fn=None, 
                        head_log_var_fn=None, **kwargs):
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
    eps = 1e-4

    input_dim_flat = input_dim #vectorize_pytree(dummy_input).shape[0]
    output_dim_flat = output_dim #[0]

    network_params = kwargs["emission_architecture"]
    init_cov_bias = network_params["cov_init"]

    # If no trunk if defined, then use the identity.
    out_trunk_features = network_params.get("out_trunk_features")
    if out_trunk_features == [] or out_trunk_features is None:
        trunk_fn = trunk_fn or Identity(input_dim)
    else:
        trunk_fn = trunk_fn or MLP(out_trunk_features)

    # If no head mean function is specified, default to an affine transformation.
    if head_mean_fn is None:
        head_mean_fn = MLP(network_params["out_mean_features"] + [output_dim,])

    # If no head mean function is specified, default to an affine transformation.
    if head_log_var_fn is None:
        # Assume a one-layer covariance function
        head_log_var_fn = nn.Dense(output_dim_flat, 
            kernel_init=nn.initializers.zeros, 
            bias_init=nn.initializers.constant(init_cov_bias))

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
            children = tuple(
                            # self.trunk_fn,
                            #  self.head_mean_fn,
                            #  self.head_log_var_fn
                             )
            aux_data = None
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

        def __call__(self, inputs):
            """
            Equivalent to the `.forward` method in PyTorch.  Generates the parameters of an independent
            multivariate Gaussian distribution as a function of the inputs.
            Args:
                inputs:
            Returns:
            """
            cov, loc = self._generate_distribution_parameters(inputs)
            return (cov, loc)

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
            loc = self.head_mean_fn(trunk_output)

            # Get the variance output and reshape it.
            var_output_flat = self.head_log_var_fn(trunk_output)
            # For the stability of training
            cov_diag = eps + softplus(var_output_flat)
            cov = np.diag(cov_diag)

            return (cov, loc)

    return NeuralNetworkEmissions(input_dim, output_dim, 
                                emissions_network=PotentialGenerator())
