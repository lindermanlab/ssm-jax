import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.base import SSM
from ssm.inference.em import em
from ssm.utils import Verbosity, ensure_has_batch_dim, auto_batch

import ssm.hmm.initial as hmm_initial
import ssm.hmm.transitions as transitions
import ssm.lds.emissions as emissions
import ssm.lds.initial as lds_initial
import ssm.slds.dynamics as dynamics
from ssm.distributions import MultivariateNormalBlockTridiag as LDSPosterior


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

    ### Methods for posterior inference
    # @format_dataset
    # def initialize(self, dataset: np.ndarray, key: jr.PRNGKey, method: str="kmeans") -> None:
    #     r"""Initialize the model parameters by performing an M-step with state assignments
    #     determined by the specified method (random or kmeans).

    #     Args:
    #         dataset (np.ndarray): array of observed data
    #             of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})`
    #         key (jr.PRNGKey): random seed
    #         method (str, optional): state assignment method.
    #             One of "random" or "kmeans". Defaults to "kmeans".

    #     Raises:
    #         ValueError: when initialize method is not recognized
    #     """
    #     # initialize assignments and perform one M-step
    #     num_states = self._num_states
    #     if method.lower() == "random":
    #         # randomly assign datapoints to clusters
    #         assignments = jr.choice(key, self._num_states, dataset.shape[:-1])

    #     elif method.lower() == "kmeans":
    #         # cluster the data with kmeans
    #         print("initializing with kmeans")
    #         from sklearn.cluster import KMeans
    #         km = KMeans(num_states)
    #         flat_dataset = dataset.reshape(-1, dataset.shape[-1])
    #         assignments = km.fit_predict(flat_dataset).reshape(dataset.shape[:-1])

    #     else:
    #         raise ValueError(f"Invalid initialize method: {method}.")

    #     # Make a dummy posterior that just exposes expected_states
    #     @dataclass
    #     class DummyPosterior:
    #         expected_states: np.ndarray
    #     dummy_posteriors = DummyPosterior(one_hot(assignments, self._num_states))

    #     # Do one m-step with the dummy posteriors
    #     self._emissions.m_step(dataset, dummy_posteriors)

    def infer_posterior(self, data):
        # TODO: fit a structured mean field posterior
        raise NotImplementedError

    ### EM: Operates on batches of data (aka datasets) and posteriors
    @ensure_has_batch_dim
    def m_step(self,
               data: np.ndarray,
               posterior: LDSPosterior,
               covariates=None,
               metadata=None,
               key: jr.PRNGKey=None):
        self._discrete_initial_condition.m_step(data, posterior.discrete_state_posterior)
        self._continuous_initial_condition.m_step(data, posterior.continuous_state_posterior)
        self._transitions.m_step(data, posterior.discrete_state_posterior)
        self._dynamics.m_step(data, posterior)
        self._emissions.m_step(data, posterior)

    # @format_dataset
    # def fit(self, dataset: np.ndarray,
    #         method: str="em",
    #         num_iters: int=100,
    #         tol: float=1e-4,
    #         initialization_method: str="kmeans",
    #         key: jr.PRNGKey=None,
    #         verbosity: Verbosity=Verbosity.DEBUG):
    #     r"""Fit the HMM to a dataset using the specified method and initialization.

    #     Args:
    #         dataset (np.ndarray): observed data
    #             of shape :math:`(\text{[batch]} , \text{num_timesteps} , \text{emissions_dim})`
    #         method (str, optional): model fit method.
    #             Must be one of ["em"]. Defaults to "em".
    #         num_iters (int, optional): number of fit iterations.
    #             Defaults to 100.
    #         tol (float, optional): tolerance in log probability to determine convergence.
    #             Defaults to 1e-4.
    #         initialization_method (str, optional): method to initialize latent states.
    #             Defaults to "kmeans".
    #         key (jr.PRNGKey, optional): Random seed.
    #             Defaults to None.
    #         verbosity (Verbosity, optional): print verbosity.
    #             Defaults to Verbosity.DEBUG.

    #     Raises:
    #         ValueError: if fit method is not reocgnized

    #     Returns:
    #         log_probs (np.ndarray): log probabilities at each fit iteration
    #         model (HMM): the fitted model
    #         posteriors (StationaryHMMPosterior): the fitted posteriors
    #     """
    #     model = self
    #     kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

    #     if initialization_method is not None:
    #         if verbosity >= Verbosity.LOUD : print("Initializing...")
    #         self.initialize(dataset, key, method=initialization_method)
    #         if verbosity >= Verbosity.LOUD: print("Done.", flush=True)

    #     if method == "em":
    #         log_probs, model, posteriors = em(model, dataset, **kwargs)
    #     else:
    #         raise ValueError(f"Method {method} is not recognized/supported.")

    #     return log_probs, model, posteriors

    def __repr__(self):
        return f"<ssm.slds.{type(self).__name__} num_states={self.num_states} " \
               f"latent_dim={self.latent_dim} " \
               f"emissions_dim={self.emissions_dim}>"
