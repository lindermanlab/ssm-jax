import warnings

import jax.numpy as np
import jax.random as jr

from ssm.factorial_hmm.posterior import FactorialHMMPosterior
from ssm.factorial_hmm.initial import FactorialInitialCondition
from ssm.factorial_hmm.transitions import FactorialTransitions
from ssm.factorial_hmm.emissions import FactorialEmissions
from ssm.hmm.base import HMM
from ssm.utils import format_dataset

class FactorialHMM(HMM):

    def __init__(self, num_states: (tuple or list),
                 initial_condition: FactorialInitialCondition,
                 transitions: FactorialTransitions,
                 emissions: FactorialEmissions):
        super().__init__(num_states, initial_condition, transitions, emissions)

    @format_dataset
    def initialize(self, dataset: np.ndarray, key: jr.PRNGKey, method: str="kmeans") -> None:
        warnings.warn(UserWarning("FactorialHMM.initialize() is not implemented!"))
        pass

    def infer_posterior(self, data):
        return FactorialHMMPosterior.infer(self._initial_condition.log_probs(data),
                                           self._emissions.log_probs(data),
                                           self._transitions.log_probs(data))
