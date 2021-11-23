import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class
from jax import lax

from ssm.lds.base import LDS
from ssm.distributions import MultivariateNormalBlockTridiag
import ssm.ctlds.initial as initial
import ssm.ctlds.dynamics as dynamics
import ssm.ctlds.emissions as emissions
from ssm.utils import Verbosity, auto_batch, ensure_has_batch_dim
CTLDSPosterior = MultivariateNormalBlockTridiag

@register_pytree_node_class
class CTLDS(LDS):
    def __init__(self,
                 initial_condition: initial.InitialCondition,
                 dynamics: dynamics.Dynamics,
                 emissions: emissions.Emissions,
                 ):
        """The CTLDS base class.

        Args:
            num_states (int): number of discrete states
            initial_condition (initial.InitialCondition):
                initial condition object defining :math:`p(z_1)`
            transitions (transitions.Transitions):
                transitions object defining :math:`p(z_t|z_{t-1})`
            emissions (emissions.Emissions):
                emissions ojbect defining :math:`p(x_t|z_t)`
        """
        super().__init__(initial_condition, dynamics, emissions)

    @property
    def drift_matrix(self):
        return self._dynamics.drift_matrix

    @property
    def drift_bias(self):
        return self._dynamics.drift_bias
    
    @property
    def diffusion_scale(self):
        return self._dynamics.diffusion_scale

    @property
    def dynamics_matrix(self):
        raise NotImplementedError

    @property
    def dynamics_bias(self):
        raise NotImplementedError

    @property
    def dynamics_noise_covariance(self):
        raise NotImplementedError

    ### Methods for posterior inference
    def initialize(self, dataset, covariates=None, metadata=None, key=None, method=None):
        """Initialize the LDS parameters.
        NOTE: Not yet implemented.
        """
        raise NotImplementedError

    @auto_batch(batched_args=("data", "covariates", "metadata", "initial_states"),
                map_function=lax.map)
    def approximate_posterior(self,
                              data,
                              covariates=None,
                              metadata=None,
                              initial_states=None):
        return laplace_approximation(self, data, initial_states)

    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior: CTLDSPosterior,
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
        self._dynamics.m_step(data, posterior, covariates)
        self._emissions.m_step(data, posterior, key=key)

    @ensure_has_batch_dim()
    def fit(self,
            data:np.ndarray,
            covariates=None,
            metadata=None,
            method: str="laplace_em",
            rng: jr.PRNGKey=None,
            num_iters: int=100,
            tol: float=1e-4,
            verbosity=Verbosity.DEBUG):
        r"""Fit the LDS to a dataset using the specified method.

        Generally speaking, we cannot perform exact EM for an LDS with arbitrary emissions.
        However, for an LDS with generalized linear model (GLM) emissions, we can perform Laplace EM.

        Args:
            data (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            method (str, optional): model fit method.
                Must be one of ["laplace_em"]. Defaults to "laplace_em".
            rng (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            elbos (np.ndarray): elbos at each fit iteration
            model (LDS): the fitted model
            posteriors (LDSPosterior): the fitted posteriors
        """
        model = self
        if method == "laplace_em":
            elbos, lds, posteriors = laplace_em(rng, model, data, num_iters=num_iters, tol=tol)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return elbos, lds, posteriors

    def __repr__(self):
        return f"<ssm.ctlds.{type(self).__name__} latent_dim={self.latent_dim} "\
            f"emissions_shape={self.emissions_shape}>"