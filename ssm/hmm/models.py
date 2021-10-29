import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax.tree_util import register_pytree_node_class

from ssm.hmm.base import HMM
from ssm.hmm.initial import StandardInitialCondition
from ssm.hmm.transitions import StationaryTransitions
from ssm.hmm.emissions import GaussianEmissions

@register_pytree_node_class
class GaussianHMM(HMM):
    def __init__(self,
                 num_states,
                 num_emission_dims,
                 initial_state_probs=None,
                 transition_matrix=None,
                 emission_means=None,
                 emission_covariances=None,
                 seed=None):

        if initial_state_probs is None:
            initial_state_probs = np.ones(num_states) / num_states

        if transition_matrix is None:
            transition_matrix = np.ones((num_states, num_states)) / num_states

        if emission_means is None:
            assert seed is not None, "You must either specify the means "
            "or give a seed (PRNGKey) so that they can be initialized randomly"

            means_prior = tfd.MultivariateNormalDiag(
                np.zeros(num_emission_dims),
                np.ones(num_emission_dims))
            emission_means = means_prior.sample(seed=seed, sample_shape=num_states)

        if emission_covariances is None:
            emission_covariances = np.tile(np.eye(num_emission_dims), (num_states, 1, 1))

        initial_condition = StandardInitialCondition(initial_probs=initial_state_probs)
        transitions = StationaryTransitions(transition_matrix=transition_matrix)
        emissions = GaussianEmissions(means=emission_means, covariances=emission_covariances)
        super(GaussianHMM, self).__init__(num_states,
                                          initial_condition,
                                          transitions,
                                          emissions)