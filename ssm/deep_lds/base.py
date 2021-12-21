import jax
from jax import grad, lax, vmap, random, jit
from jax.ops import index, index_add
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_pytree_node_class
import jax.numpy as np
import jax.random as jr

from functools import partial

from ssm.base import SSM
from ssm.lds.base import LDS
from ssm.lds.initial import StandardInitialCondition
from ssm.lds.dynamics import StationaryDynamics
from ssm.utils import Verbosity, ensure_has_batch_dim, auto_batch, random_rotation

from ssm.inference.deep_vi import deep_variational_inference
from ssm.lds_svae.posterior import LDSSVAEPosterior, DKFPosterior
from ssm.nn_util import build_gaussian_network

# To keep it really really simple, we don't even to write ANYTHING
# Except for the fit function
@register_pytree_node_class
class DeepLDS(LDS):

    def __init__(self, num_latent_dims,
                 num_emission_dims,
                 emissions,
                 initial_state_mean: np.ndarray=None,
                 initial_state_scale_tril: np.ndarray=None,
                 dynamics_weights: np.ndarray=None,
                 dynamics_bias: np.ndarray=None,
                 dynamics_scale_tril: np.ndarray=None,
                 seed: jr.PRNGKey=None
                 ):
        
        if initial_state_mean is None:
            initial_state_mean = np.zeros(num_latent_dims)

        if initial_state_scale_tril is None:
            initial_state_scale_tril = np.eye(num_latent_dims)

        if dynamics_weights is None:
            seed, rng = jr.split(seed, 2)
            dynamics_weights = random_rotation(rng, num_latent_dims, theta=np.pi/20)

        if dynamics_bias is None:
            dynamics_bias = np.zeros(num_latent_dims)

        if dynamics_scale_tril is None:
            dynamics_scale_tril = 0.1**2 * np.eye(num_latent_dims)

        initial_condition = StandardInitialCondition(initial_mean=initial_state_mean,
                                                     initial_scale_tril=initial_state_scale_tril)
        transitions = StationaryDynamics(weights=dynamics_weights,
                                         bias=dynamics_bias,
                                         scale_tril=dynamics_scale_tril)

        super(DeepLDS, self).__init__(initial_condition, transitions, emissions)

    @property
    def latent_dim(self):
        return self._emissions.latent_dim

    @property
    def emissions_network(self):
        return self._emissions
    
    @property
    def emissions_shape(self):
        return self._emissions.emissions_shape

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(*children)
        return obj

    # This is identical to Collin's original code for LDS
    # Except that it returns  
    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior,
               covariates=None,
               metadata=None,
               key: jr.PRNGKey=None):
        """Update the model in a (potentially approximate) M step.
        
        Updates the model in place.
        Args:
            data (np.ndarray): observed data with shape (B, T, D)  
            posterior (LDSPosterior): LDS posterior object with leaf shapes (B, ...).
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            key (jr.PRNGKey, optional): random seed. Defaults to None.
        """
        # self._initial_condition.m_step(dataset, posteriors)  # TODO initial dist needs prior
        self._dynamics.m_step(data, posterior)
        self._emissions.m_step(data, posterior, key=key)
        return self

    @ensure_has_batch_dim()
    def fit(self,
            key: jr.PRNGKey,
            data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="svae",
            num_iters: int=100,
            tol: float=1e-4,
            verbosity: Verbosity=Verbosity.DEBUG
            # Should have an option of providing a recognition net architecture 
            ):
        """
        Notably, returns the rec net as well as the model
        """
        # Just initialize the posterior since the model will be
        # updated on the first m-step.
        
        # TODO: figure out the proper assumptions of data shape!
        N, T, D = data.shape

        if method == "svae":
            posterior = LDSSVAEPosterior.initialize(
            self, data, covariates=covariates, metadata=metadata)
            rec_net = build_gaussian_network(D, self.latent_dim)
        elif method == "dkf":
            posterior = DKFPosterior.initialize(
            self, data, covariates=covariates, metadata=metadata)
            rec_net = Bidirectional_RNN.from_params(self.latent_dim)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        bounds, model, posterior = deep_variational_inference(
                key, self, data, rec_net, posterior, covariates=covariates, metadata=metadata,
                num_iters=num_iters, tol=tol, verbosity=verbosity)

        return bounds, model, posterior
