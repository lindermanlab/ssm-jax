import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp

from ssm.base import SSM
import ssm.hmm.initial as hmm_initial
import ssm.hmm.transitions as transitions
from ssm.inference.variational_em import variational_em
import ssm.lds.initial as lds_initial
import ssm.slds.emissions as emissions
import ssm.slds.dynamics as dynamics
from ssm.slds.posterior import StructuredMeanFieldSLDSPosterior
from ssm.utils import Verbosity, ensure_has_batch_dim

tfd = tfp.distributions

@register_pytree_node_class
class SLDS(SSM):
    """Base class for switching linear dynamical systems.
    """
    def __init__(self,
                 num_states: int,
                 latent_dim: int,
                 discrete_initial_condition: hmm_initial.InitialCondition,
                 continuous_initial_condition: lds_initial.InitialCondition,
                 transitions: transitions.Transitions,
                 dynamics: dynamics.Dynamics,
                 emissions: emissions.Emissions,
                 ):
        """The SLDS base class.

        Args:
            num_states (int): number of discrete states
            latent_dim (int): number of continuous latent dimensions
            discrete_initial_condition (hmm.initial.InitialCondition):
                initial condition object defining :math:`p(z_1)`
            continuous_initial_condition (lds.initial.InitialCondition):
                initial condition object defining :math:`p(x_1)`
            transitions (transitions.Transitions):
                transitions object defining :math:`p(z_t | z_{t-1})`
            dynamics (dynamics.Dynamics):
                dynamics object defining :math:`p(x_t | x_{t-1}, z_t)`
            emissions (emissions.Emissions):
                emissions ojbect defining :math:`p(y_t | x_t, z_t)`
        """
        # Store these as private class variables
        self._num_states = num_states
        self._latent_dim = latent_dim
        self._discrete_initial_condition = discrete_initial_condition
        self._continuous_initial_condition = continuous_initial_condition
        self._transitions = transitions
        self._dynamics = dynamics
        self._emissions = emissions

    @property
    def transition_matrix(self):
        return self._transitions.transition_matrix

    @property
    def num_states(self):
        return self._num_states

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def emissions_shape(self):
        return self._emissions.emissions_shape

    def tree_flatten(self):
        children = (self._discrete_initial_condition,
                    self._continuous_initial_condition,
                    self._transitions,
                    self._dynamics,
                    self._emissions)
        aux_data = (self._num_states, self._latent_dim)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # We have to be a little fancy since this classmethod
        # is inherited by subclasses with different constructors.
        obj = object.__new__(cls)
        SLDS.__init__(obj, *aux_data, *children)
        return obj

    def initial_distribution(self, covariates=None, metadata=None):
        return tfd.JointDistributionNamed(dict(
            discrete=self._discrete_initial_condition.distribution(
                covariates=covariates, metadata=metadata),
            continuous=self._continuous_initial_condition.distribution(
                covariates=covariates, metadata=metadata)
            ))

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        z_prev = state["discrete"]
        x_prev = state["continuous"]
        return tfd.JointDistributionNamed(dict(
            discrete=self._transitions.distribution(
                z_prev, covariates=covariates, metadata=metadata),
            continuous=lambda discrete: self._dynamics.distribution(
                x_prev, discrete, covariates=covariates, metadata=metadata)
        ))

    def emissions_distribution(self, state, covariates=None, metadata=None):
        return self._emissions.distribution(state)

    ### EM: Operates on batches of data (aka datasets) and posteriors
    @ensure_has_batch_dim()
    def m_step(self,
               data: np.ndarray,
               posterior, #: SLDSPosterior,
               covariates=None,
               metadata=None,
               key: jr.PRNGKey=None):
        self._discrete_initial_condition.m_step(data, posterior.discrete_posterior, covariates=covariates, metadata=metadata)
        self._continuous_initial_condition.m_step(data, posterior.continuous_posterior, covariates=covariates, metadata=metadata)
        self._transitions.m_step(data, posterior.discrete_posterior, covariates=covariates, metadata=metadata)
        self._dynamics.m_step(data, posterior, covariates=covariates, metadata=metadata)
        self._emissions.m_step(data, posterior, covariates=covariates, metadata=metadata, key=key)
        return self

    @ensure_has_batch_dim()
    def fit(self,
            key: jr.PRNGKey,
            data: np.ndarray,
            covariates=None,
            metadata=None,
            method: str="variational_em",
            num_iters: int=100,
            tol: float=1e-4,
            verbosity: Verbosity=Verbosity.DEBUG,
            callback=None):
        r"""Fit the HMM to a dataset using the specified method and initialization.

        Args:
            data (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
            method (str, optional): model fit method.
                Must be one of ["em"]. Defaults to "em".
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            key (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            bounds (np.ndarray): log probabilities at each fit iteration
            model (SLDS): the fitted model
            posterior (SLDSPosterior): the fitted posteriors
        """
        # Just initialize the posterior since the model will be
        # updated on the first m-step.
        posterior = StructuredMeanFieldSLDSPosterior.initialize(
            self, data, covariates=covariates, metadata=metadata)

        if method == "variational_em":
            return variational_em(
                key, self, data, posterior, covariates=covariates, metadata=metadata,
                num_iters=num_iters, tol=tol, verbosity=verbosity, callback=callback)

        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

    def __repr__(self):
        return f"<ssm.slds.{type(self).__name__} num_states={self.num_states} " \
               f"latent_dim={self.latent_dim} " \
               f"emissions_shape={self.emissions_shape}>"
