from ssm.models import SSM
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
import copy

from ssm.inference.utils import compute_expected_suff_stats
from .base import _StandardLDS


class GaussianLDS(_StandardLDS):
    def __init__(self, observation_dim, hidden_dim):
        super(GaussianLDS, self).__init__(observation_dim, hidden_dim)

        self.transition_matrix = None  # A_t
        self.emission_matrix = None  # B_t
        self.transition_scale_tril = None  # \Sigma_t^h
        self.emission_scale_tril = None  # \Sigma_t^v
        self.hidden_bias = None  # \bar{h_t}
        self.output_bias = None  # \bar{v_t}

        self.initial_mean = None  # \mu_\pi
        self.initial_scale_tril = None  # \Sigma_\pi

    def initial_dist(self):
        return tfp.distributions.MultivariateNormalTriL(
            loc=self.initial_mean, scale_tril=self.initial_scale_tril
        )

    def dynamics_dist(self, state):
        loc = self.transition_matrix @ state + self.hidden_bias
        return tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=self.transition_scale_tril
        )

    def emissions_dist(self, state):
        loc = self.emission_matrix @ state + self.output_bias
        return tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=self.emission_scale_tril
        )

    @property
    def initial_state_probs(self):
        return self._initial_dist.probs_parameter()
