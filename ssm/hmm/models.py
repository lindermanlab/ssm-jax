import inspect
import jax.numpy as np
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax.tree_util import register_pytree_node_class

import ssm.distributions as ssmd
from ssm.hmm.base import HMM
from ssm.hmm.initial import StandardInitialCondition
from ssm.hmm.transitions import Transitions, StationaryTransitions
from ssm.hmm.emissions import BernoulliEmissions, GaussianEmissions, PoissonEmissions
from ssm.utils import Verbosity, random_rotation, make_named_tuple, ensure_has_batch_dim, auto_batch

import warnings

@register_pytree_node_class
class BernoulliHMM(HMM):
    def __init__(self,
                 num_states: int,
                 num_emission_dims: int=None,
                 initial_state_probs: np.ndarray=None,
                 transition_matrix: np.ndarray=None,
                 emission_probs: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        r"""HMM with conditionally independent Bernoulli emissions.

        .. math::
            p(x_t | z_t = k) \sim \prod_{d=1}^D \mathrm{Bern}(x_{td} \mid p_{kd})

        where :math:`p_{kd}` is the probability of seeing a one in dimension :math:`d`
        given discrete latent state :math:`k`.

        The BernoulliHMM can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_states``, ``num_emission_dims``, and ``seed``
        to create a BernoulliHMM with generic, randomly initialized parameters.

        Args:
            num_states (int): number of discrete latent states
            num_emission_dims (int, optional): number of emission dims. Defaults to None.
            initial_state_probs (np.ndarray, optional): initial state probabilities
                with shape :math:`(\text{num\_states},)`. Defaults to None.
            transition_matrix (np.ndarray, optional): transition matrix
                with shape :math:`(\text{num\_states}, \text{num\_states})`. Defaults to None.
            emission_probs] (np.ndarray, optional): specifies emission probabilities
                with shape :math:`(\text{num\_states}, \text{emission\_dims})`. Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """

        if initial_state_probs is None:
            initial_state_probs = np.ones(num_states) / num_states

        if transition_matrix is None:
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if emission_probs is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_means or give a dimension and seed (PRNGKey) "\
                "so that they can be initialized randomly."

            probs_prior = ssmd.Beta(1, 1)
            emission_probs = probs_prior.sample(seed=seed, sample_shape=(num_states, num_emission_dims))

        # Grab the parameter values.  This allows us to explicitly re-build the object.
        self._parameters = make_named_tuple(dict_in=locals(),
                                            keys=list(inspect.signature(self.__init__)._parameters.keys()),
                                            name=str(self.__class__.__name__) + 'Tuple')

        initial_condition = StandardInitialCondition(num_states, initial_probs=initial_state_probs)
        transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)
        emissions = BernoulliEmissions(num_states, probs=emission_probs)
        super(BernoulliHMM, self).__init__(num_states,
                                           initial_condition,
                                           transitions,
                                           emissions)


@register_pytree_node_class
class GaussianHMM(HMM):
    """HMM with Gaussian emissions.
    """
    def __init__(self,
                 num_states: int,
                 num_emission_dims: int=None,
                 initial_state_probs: np.ndarray=None,
                 transitions: Transitions=None,
                 transition_matrix: np.ndarray=None,
                 emission_means: np.ndarray=None,
                 emission_covariances: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        r"""HMM with Gaussian emissions.

        .. math::
            p(x_t | z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)

        The GaussianHMM can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_states``, ``num_emission_dims``, and ``seed``
        to create a GaussianHMM with generic, randomly initialized parameters.

        Args:
            num_states (int): number of discrete latent states
            num_emission_dims (int, optional): number of emission dims. Defaults to None.
            initial_state_probs (np.ndarray, optional): initial state probabilities
                with shape :math:`(\text{num\_states},)`. Defaults to None.
            transitions (Transitions, optional): object specifying transitions
                Defaults to None. If specified, then `transition_matrix` is ignored.
            transition_matrix (np.ndarray, optional): transition matrix
                with shape :math:`(\text{num\_states}, \text{num\_states})`.
                Defaults to None. Only used if `transitions` is None.
            emission_means (np.ndarray, optional): specifies emission means
                with shape :math:`(\text{num\_states}, \text{emission\_dims})`. Defaults to None.
            emission_covariances (np.ndarray, optional): specifies emissions covariances
                with shape :math:`(\text{num\_states}, \text{emission\_dims}, \text{emission\_dims})`.
                Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """

        if initial_state_probs is None:
            initial_state_probs = np.ones(num_states) / num_states

        if (transitions is None) and (transition_matrix is None):
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if transitions is None:
            transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)

        if emission_means is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_means or give a dimension and seed (PRNGKey) "\
                "so that they can be initialized randomly."

            means_prior = tfd.MultivariateNormalDiag(
                np.zeros(num_emission_dims),
                np.ones(num_emission_dims))
            emission_means = means_prior.sample(seed=seed, sample_shape=num_states)

        if emission_covariances is None:
            assert num_emission_dims is not None, \
                "You must either specify the emission_covariances or give a dimension "\
                "so that they can be initialized."
            emission_covariances = np.tile(np.eye(num_emission_dims), (num_states, 1, 1))

        initial_condition = StandardInitialCondition(num_states, initial_probs=initial_state_probs)
        emissions = GaussianEmissions(num_states, means=emission_means, covariances=emission_covariances)
        super(GaussianHMM, self).__init__(num_states,
                                          initial_condition,
                                          transitions,
                                          emissions)


@register_pytree_node_class
class PoissonHMM(HMM):
    def __init__(self,
                 num_states: int,
                 num_emission_dims: np.ndarray=None,
                 initial_state_probs: np.ndarray=None,
                 transition_matrix: np.ndarray=None,
                 emission_rates: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        r"""HMM with Poisson emissions.

        .. math::
            p(x_t | z_t = k) \sim \text{Po}(\lambda=\lambda_k)

        The PoissonHMM can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_states``, ``num_emission_dims``, and ``seed``
        to create a GaussianHMM with generic, randomly initialized parameters.

        Args:
            num_states (int): number of discrete latent states
            num_emission_dims (int, optional): number of emission dims. Defaults to None.
            initial_state_probs (np.ndarray, optional): initial state probabilities
                with shape :math:`(\text{num\_states},)`. Defaults to None.
            transition_matrix (np.ndarray, optional): transition matrix
                with shape :math:`(\text{num\_states}, \text{num\_states})`. Defaults to None.
            emission_rates (np.ndarray, optional): specifies Poisson emission rates
                with shape :math:`(\text{num\_states}, \text{emission\_dims})`. Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """

        if initial_state_probs is None:
            initial_state_probs = np.ones(num_states) / num_states

        if transition_matrix is None:
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if emission_rates is None:
            assert seed is not None and num_emission_dims is not None, \
                "You must either specify the emission_rates " \
                "or give the num_emission_dims and a seed (PRNGKey) "\
                "so that they can be initialized randomly"

            means_prior = tfp.distributions.Gamma(3.0, 1.0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                emission_rates = means_prior.sample(seed=seed, sample_shape=(num_states, num_emission_dims))

        initial_condition = StandardInitialCondition(num_states, initial_probs=initial_state_probs)
        transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)
        emissions = PoissonEmissions(num_states, rates=emission_rates)
        super(PoissonHMM, self).__init__(num_states,
                                         initial_condition,
                                         transitions,
                                         emissions)

