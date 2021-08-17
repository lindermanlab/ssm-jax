from ssm.models import SSM
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
import copy

from ssm.inference.utils import compute_expected_suff_stats


class _StandardLDS(SSM):
    def __init__(self, observation_dim, hidden_dim):
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
