import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class
from jax import lax

from ssm.base import SSM
from ssm.inference.laplace_em import laplace_approximation, laplace_em
import ssm.lds.initial as initial
import ssm.lds.dynamics as dynamics
import ssm.lds.emissions as emissions
from ssm.utils import Verbosity, auto_batch, ensure_has_batch_dim

from ssm.distributions import MultivariateNormalBlockTridiag
LDSPosterior = MultivariateNormalBlockTridiag

@register_pytree_node_class
class LDS(SSM):
    def __init__(self,
                 initial_condition: initial.InitialCondition,
                 dynamics: dynamics.Dynamics,
                 emissions: emissions.Emissions,
                 ):
        """The LDS base class.

        Args:
            num_states (int): number of discrete states
            initial_condition (initial.InitialCondition):
                initial condition object defining :math:`p(z_1)`
            transitions (transitions.Transitions):
                transitions object defining :math:`p(z_t|z_{t-1})`
            emissions (emissions.Emissions):
                emissions ojbect defining :math:`p(x_t|z_t)`
        """
        self._initial_condition = initial_condition
        self._dynamics = dynamics
        self._emissions = emissions

    @property
    def latent_dim(self):
        return self._emissions.weights.shape[-1]

    @property
    def emissions_shape(self):
        return self._emissions.emissions_shape
    @property
    def initial_mean(self):
        return self._initial_condition.mean

    @property
    def initial_covariance(self):
        return self._initial_condition._distribution.covariance()

    @property
    def dynamics_matrix(self):
        return self._dynamics.weights

    @property
    def dynamics_bias(self):
        return self._dynamics.bias

    @property
    def dynamics_noise_covariance(self):
        Q_sqrt = self._dynamics.scale_tril  # TODO: this should be a property in dynamics
        return Q_sqrt @ Q_sqrt.T

    @property
    def emissions_matrix(self):
        return self._emissions.weights

    @property
    def emissions_bias(self):
        return self._emissions.bias

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)

    def initial_distribution(self, covariates=None, metadata=None):
        return self._initial_condition.distribution(covariates=covariates, metadata=metadata)

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        return self._dynamics.distribution(state, covariates=covariates, metadata=metadata)

    def emissions_distribution(self, state, covariates=None, metadata=None):
        return self._emissions.distribution(state, covariates=covariates, metadata=metadata)

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
               posterior: LDSPosterior,
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
        return f"<ssm.lds.{type(self).__name__} latent_dim={self.latent_dim} "\
            f"emissions_shape={self.emissions_shape}>"