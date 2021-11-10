from ssm.hmm.base import HMM
import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.factorial_hmm.posterior import FactorialHMMPosterior
from ssm.factorial_hmm.initial import FactorialInitialCondition
from ssm.factorial_hmm.transitions import FactorialTransitions
from ssm.factorial_hmm.emissions import FactorialEmissions, NormalFactorialEmissions
from ssm.hmm.base import HMM
from ssm.hmm.transitions import StationaryTransitions
from ssm.hmm.initial import StandardInitialCondition
from ssm.utils import format_dataset


class FactorialHMM(HMM):

    def __init__(self, num_states: (tuple or list),
                 initial_condition: FactorialInitialCondition,
                 transitions: FactorialTransitions,
                 emissions: FactorialEmissions):
        super().__init__(num_states, initial_condition, transitions, emissions)

    @format_dataset
    def initialize(self, dataset: np.ndarray, key: jr.PRNGKey, method: str="kmeans") -> None:
        pass

    def infer_posterior(self, data):
        return FactorialHMMPosterior.infer(self._initial_condition.log_probs(data),
                                           self._emissions.log_probs(data),
                                           self._transitions.log_probs(data))


@register_pytree_node_class
class NormalFactorialHMM(FactorialHMM):

    def __init__(self,
                 num_states: (tuple or list),
                 initial_probs: (tuple or list)=None,
                 transition_matrices: (tuple or list)=None,
                 emission_means: (tuple or list)=None,
                 emission_variance: float=0.1**2,
                 seed: jr.PRNGKey=None):
        r"""
        Factorial HMM with scalar Gaussian emissions.

        The model consists of :math:`G` discrete latent states
        :math:`z_t = (z_{t1}, \ldots, z_{tG})`. The :math:`g`-th state takes
        values :math:`(0, ..., K_g-1)`.

        The emission mean is a sum of means associated with each group,
        ..math:
            \mathbb{E}[x_t \mid z_t ] = \sum_g \mu_{g,z_{tg}}

        For example, the emissions may be measurements from a home's power
        meter and the groups may correspond to appliances, which may be either
        on or off (:math:`z_{tg} \in \{0,1\}`). If off, the appliance contributes
        nothing to the measurement (:math:`\mu_{g,0}=0`), but if on, the
        appliance contributes :math:`\mu_{g,1}`.

        Args:
            num_states (tuple or list): number of discrete latent states per group
            initial_state_probs (np.ndarray, optional): initial state probabilities
                with shape :math:`(\text{num_states},)`. Defaults to None.
            transitions (Transitions, optional): object specifying transitions
                Defaults to None. If specified, then `transition_matrix` is ignored.
            transition_matrix (np.ndarray, optional): transition matrix
                with shape :math:`(\text{num_states}, \text{num_states})`.
                Defaults to None. Only used if `transitions` is None.
            emission_means (np.ndarray, optional): specifies emission means
                with shape :math:`(\text{num_states}, \text{emission_dims})`. Defaults to None.
            emission_covariances (np.ndarray, optional): specifies emissions covariances
                with shape :math:`(\text{num_states}, \text{emission_dims}, \text{emission_dims})`.
                Defaults to None.
            seed (jr.PRNGKey, optional): random seed. Defaults to None.
        """
        assert isinstance(num_states, (tuple, list))
        for K in num_states:
            assert isinstance(K, int)

        if initial_probs is None:
            initial_conditions = tuple(
                StandardInitialCondition(K, np.ones(K) / K) for K in num_states
            )

        if transition_matrices is None:
            transmat = lambda K: 0.9 * np.eye(K) + 0.1 / (K-1) * (1 - np.eye(K))
            transitions = tuple(
                StationaryTransitions(K, transmat(K)) for K in num_states
            )

        if emission_means is None:
            assert seed is not None, \
                "You must either specify the emission_means or give a dimension and seed (PRNGKey) "\
                "so that they can be initialized randomly."
            emission_means = []
            for K in num_states:
                this_seed, seed = jr.split(seed, 2)
                emission_means.append(tfd.Normal(0, 1).sample(seed=this_seed, sample_shape=K))

        factorial_initial_condition = FactorialInitialCondition(initial_conditions)
        factorial_transitions = FactorialTransitions(transitions)
        factorial_emissions = NormalFactorialEmissions(
            num_states, means=emission_means, log_scale=np.log(np.sqrt(emission_variance)))
        super(NormalFactorialHMM, self).__init__(
            num_states, factorial_initial_condition, factorial_transitions, factorial_emissions)

