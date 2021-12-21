import jax
from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.distributions import GaussianLinearRegression, glm
from ssm.lds.emissions import Emissions
from ssm.nn_util import build_gaussian_network

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
            self._emissions_network = build_gaussian_network(in_dim, out_dim)
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
        scale_diag, loc = self._emissions_network.apply(self._params, state)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)


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

