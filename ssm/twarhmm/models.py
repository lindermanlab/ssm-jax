"""
Model classes for time warped ARHMMs.
"""
from collections import namedtuple

import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.arhmm.base import AutoregressiveHMM
from ssm.distributions.linreg import GaussianLinearRegressionPrior
from ssm.factorial_hmm.base import FactorialHMM
from ssm.factorial_hmm.initial import FactorialInitialCondition
from ssm.factorial_hmm.transitions import FactorialTransitions
from ssm.hmm.transitions import StationaryTransitions, SimpleStickyTransitions
from ssm.twarhmm.emissions import TimeWarpedAutoregressiveEmissions
from ssm.utils import ensure_has_batch_dim


@register_pytree_node_class
class GaussianTWARHMM(FactorialHMM, AutoregressiveHMM):
    def __init__(self,
                 num_discrete_states: int,
                 time_constants: np.ndarray,
                 num_emission_dims: int=None,
                 initial_state_probs: tuple=None,
                 discrete_state_transition_matrix: np.ndarray=None,
                 time_constant_stay_probability: float=0.98,
                 emission_weights: np.ndarray=None,
                 emission_biases: np.ndarray=None,
                 emission_covariances: np.ndarray=None,
                 emission_prior: GaussianLinearRegressionPrior=None,
                 seed: jr.PRNGKey=None):
        """Gaussian time-warped autoregressive HMM.
        
        This model has the following likelihood,

        .. math:
            x_t - x_{t-1} \mid x_{t-1}, z_t \sim \mathcal{N}\left(\tau_t^{-1} (A_{z_t} x_{t-1} + b_{z_t}), \tau_t^{-1} Q_{z_t} \right).

        When the time constant is small, dynamics are faster. When the time constant is large, dynamics are slower.

        Technically, the time-warped ARHMM is a special case of a _factorial_ HMM, 
        since we model the time-constants as taking values in a discrete set and following
        a Markov transition model,

        .. math:
            p(z_t, \tau_t \mid z_{t-1}, \tau_{t-1}) = p(z_t \mid z_{t-1}) \times p(\tau_t \mid \tau_{t-1})

        Args:
            num_discrete_states (int): number of discrete latent states
            time_constants (np.ndarray): discrete non-negative values for the possible time constants.
            num_emission_dims (int, optional): emissions dimensionality. Defaults to None.
            initial_state_probs (np.ndarray, optional): initial discrete state probabilities
                with shape :math:`(\text{num\_discrete\_states},)`. Defaults to None.
            discrete_state_transition_matrix (np.ndarray, optional): transition matrix for discrete states
                with shape :math:`(\text{num\_discrete\_states}, \text{num\_discrete\_states})`.
                Defaults to None.
            time_constant_stay_probability (float, optional): defines transition matrix for time constants.
                Probability of using the same time constant as in the previous time step. Other transition
                probabilities are calculated uniformly. Defaults to 0.98.
            emission_weights (np.ndarray, optional): emission weights ..math`A_{z_t}` with shape
                :math:`(\text{num\_discrete\_states}, \text{emissions\_dim}, \text{emissions\_dim} * \text{num\_lags})`.
                Defaults to None.
            emission_biases (np.ndarray, optional): emission biases ..math`b_{z_t}`
                with shape :math:`(\text{num\_discrete\_states}, \text{emissions\_dim})`. Defaults to None.
            emission_covariances (np.ndarray, optional): emission covariance ..math`Q_{z_t}`
                with shape :math:`(\text{num\_discrete\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Defaults to None.
            emission_prior (GaussianLinearRegressionPrior, optional): prior on emissions. Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """

        assert time_constants.ndim == 1
        assert time_constants.min() > 0
        self.time_constants = time_constants

        num_time_constants = len(time_constants)
        num_states = (num_discrete_states, num_time_constants)

        # Initialize the initial state distribution
        if initial_state_probs is None:
            initial_state_probs = tuple(np.ones(K) / K for K in num_states)
        initial_condition = FactorialInitialCondition(initial_probs=initial_state_probs)

        # Initialize the discrete state transitions
        if discrete_state_transition_matrix is None:
            discrete_state_transition_matrix = 0.9 * np.eye(num_discrete_states) + \
                0.1 / (num_discrete_states - 1) * (1 - np.eye(num_discrete_states))
        else:
            assert discrete_state_transition_matrix.shape == (num_discrete_states, num_discrete_states)
            assert np.allclose(np.sum(discrete_state_transition_matrix, axis=1), 1.0)
        discrete_state_transitions = \
                StationaryTransitions(num_discrete_states, discrete_state_transition_matrix)

        # Initialize the time constant transitions
        time_constant_transitions = \
            SimpleStickyTransitions(num_time_constants, time_constant_stay_probability)

        transitions = FactorialTransitions((discrete_state_transitions, time_constant_transitions))

        # Initialize the emissions
        if emission_weights is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_weights or give a dimension "\
                "and seed (PRNGKey) so that they can be initialized randomly."
            this_seed, seed = jr.split(seed, 2)
            emission_weights = tfd.Normal(0, 0.1).sample(
                seed=this_seed,
                sample_shape=(num_discrete_states, num_emission_dims, num_emission_dims))

        if emission_biases is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_weights or give a dimension, "\
                "number of lags, and seed (PRNGKey) so that they can be initialized randomly."
            this_seed, seed = jr.split(seed, 2)
            emission_biases = tfd.Normal(0, .1).sample(
                seed=this_seed,
                sample_shape=(num_discrete_states, num_emission_dims))

        if emission_covariances is None:
            assert num_emission_dims is not None, \
                "You must either specify the emission_covariances or give a dimension "\
                "so that they can be initialized."
            emission_covariances = np.tile(np.eye(num_emission_dims), (num_discrete_states, 1, 1))

        emissions = TimeWarpedAutoregressiveEmissions(
            num_discrete_states,
            time_constants,
            weights=emission_weights,
            biases=emission_biases,
            scale_trils=np.linalg.cholesky(emission_covariances),
            emissions_distribution_prior=emission_prior)

        super().__init__(num_states, initial_condition, transitions, emissions)

    @property
    def num_discrete_states(self):
        return self.num_states[0]

    @property
    def num_time_constants(self):
        return self.num_states[1]

    @property
    def emission_weights(self):
        return self._emissions._weights

    @property
    def emission_biases(self):
        return self._emissions._biases

    @property
    def emission_scale_trils(self):
        return self._emissions._scale_trils

    @ensure_has_batch_dim()
    def initialize(self,
                   key: jr.PRNGKey,
                   data: np.ndarray,
                   covariates=None,
                   metadata=None,
                   method: str="kmeans") -> None:
        """Initialize the model parameters by performing an M-step with state assignments
        determined by the specified method (random or kmeans).

        Args:
            dataset (np.ndarray): array of observed data
                of shape :math:`(\text{batch} , \text{num\_timesteps} , \text{emissions\_dim})`
            key (jr.PRNGKey): random seed
            method (str, optional): state assignment method.
                One of "random" or "kmeans". Defaults to "kmeans".

        Raises:
            ValueError: when initialize method is not recognized
        """
        num_batches, num_timesteps = data.shape[:2]

        # initialize assignments and perform one M-step
        if method.lower() == "kmeans":
            # cluster the data with kmeans
            # print("initializing with kmeans")
            from sklearn.cluster import KMeans
            km = KMeans(self.num_discrete_states)
            flat_dataset = data.reshape(num_batches * num_timesteps, -1)
            assignments = km.fit_predict(flat_dataset).reshape(num_batches, num_timesteps)

        else:
            raise ValueError(f"Invalid initialize method: {method}.")

        # Make a dummy posterior that just exposes expected_states
        default_time_const = np.argmin((self.time_constants - 1.0)**2)
        expected_states = np.zeros((num_batches, num_timesteps) + self.num_states)
        for i, batch_assignments in enumerate(assignments):
            expected_states = expected_states.at[
                i, np.arange(num_timesteps), batch_assignments, default_time_const].set(1)

        DummyPosterior = namedtuple("DummyPosterior", ["expected_states"])
        dummy_posteriors = DummyPosterior(expected_states)

        # Do one m-step with the dummy posteriors
        self._emissions = self._emissions.m_step(data, dummy_posteriors)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    # directly GaussianHMM using parent (HMM) constructor
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(aux_data, *children)
        return obj
