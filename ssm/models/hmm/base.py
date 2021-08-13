from ssm.models import SSM
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
import copy

from ssm.inference.utils import compute_expected_suff_stats


class _StandardHMM(SSM):
    def __init__(
        self,
        num_states,
        initial_state_probs=None,
        initial_state_logits=None,
        initial_state_prior_concentration=1.0,
        transition_matrix=None,
        transition_logits=None,
        transition_prior_concentration=1.0,
    ):
        self.num_states = num_states

        # Set up the initial state distribution and prior
        if initial_state_probs is None and initial_state_logits is None:
            initial_state_logits = np.zeros(num_states)

        self._initial_dist = tfp.distributions.Categorical(
            logits=initial_state_logits, probs=initial_state_probs
        )

        self._initial_state_prior = tfp.distributions.Dirichlet(
            initial_state_prior_concentration * np.ones(num_states)
        )

        # Set up the transition matrix and prior
        if transition_matrix is None and transition_logits is None:
            transition_logits = np.zeros((num_states, num_states))
            
        self._dynamics_dist = tfp.distributions.Categorical(
            logits=transition_logits, probs=transition_matrix
        )

        self._dynamics_prior = tfp.distributions.Dirichlet(
            transition_prior_concentration * np.ones((num_states, num_states))
        )

    def initial_dist(self):
        return self._initial_dist

    def dynamics_dist(self, state):
        return self._dynamics_dist[state]

    @property
    def initial_state_probs(self):
        return self._initial_dist.probs_parameter()

    @property
    def transition_matrix(self):
        return self._dynamics_dist.probs_parameter()
