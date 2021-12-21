import warnings

import jax.numpy as np
import jax.random as jr

from ssm.factorial_hmm.posterior import FactorialHMMPosterior
from ssm.factorial_hmm.initial import FactorialInitialCondition
from ssm.factorial_hmm.transitions import FactorialTransitions
from ssm.factorial_hmm.emissions import FactorialEmissions
from ssm.hmm.base import HMM
from ssm.utils import ensure_has_batch_dim, auto_batch

class FactorialHMM(HMM):

    def __init__(self, num_states: (tuple or list),
                 initial_condition: FactorialInitialCondition,
                 transitions: FactorialTransitions,
                 emissions: FactorialEmissions):
        r"""
        Factorial HMM base class.

        The model consists of :math:`G` discrete latent states
        :math:`z_t = (z_{t1}, \ldots, z_{tG})`. The :math:`g`-th state takes
        values :math:`(0, ..., K_g-1)`.

        Args:
            num_states (tuple or list): number of discrete latent states per group
            initial_condition (FactorialInitialCondition): factorial initial state object
            transitions (FactorialTransitions): factorial transitions object
            emissions (FactorialEmissions): factorial emissions object
        """
        super().__init__(num_states, initial_condition, transitions, emissions)

    @ensure_has_batch_dim()
    def initialize(self,
                   data: np.ndarray,
                   covariates: np.ndarray=None,
                   metadata=None,
                   key: jr.PRNGKey=None,
                   method: str="kmeans") -> None:
        warnings.warn(UserWarning("FactorialHMM.initialize() is not implemented!"))
        pass

    @auto_batch(batched_args=("data", "covariates", "metadata"))
    def e_step(self, data, covariates=None, metadata=None):
        return FactorialHMMPosterior.infer(
            self._initial_condition.log_initial_probs(data, covariates=covariates, metadata=metadata),
            self._emissions.log_likelihoods(data, covariates=covariates, metadata=metadata),
            self._transitions.log_transition_matrices(data, covariates=covariates, metadata=metadata))
