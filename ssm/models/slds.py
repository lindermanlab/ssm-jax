import jax.numpy as np
import jax.nn as nn
from jax.tree_util import register_pytree_node_class

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.models.base import SSM


@register_pytree_node_class
class GaussianSLDS(SSM):
    def __init__(self,
                 initial_state_logits=None,
                 transition_logits=None,
                 initial_means=None,
                 initial_scale_trils=None,
                 dynamics_matrices=None,
                 dynamics_biases=None,
                 dynamics_scale_trils=None,
                 emissions_matrices=None,
                 emissions_biases=None,
                 emissions_scale_trils=None):
        """ TODO
        """
        self.initial_state_logits = initial_state_logits
        self.transition_logits = transition_logits
        self.initial_means = initial_means
        self.initial_scale_trils = initial_scale_trils
        self.dynamics_matrices = dynamics_matrices
        self.dynamics_biases = dynamics_biases
        self.dynamics_scale_trils = dynamics_scale_trils
        self.emissions_matrices = emissions_matrices
        self.emissions_biases = emissions_biases
        self.emissions_scale_trils = emissions_scale_trils

    @property
    def latent_dim(self):
        return self._dynamics_matrices[0].shape[0]

    @property
    def emissions_dim(self):
        return self._emissions_weights[-1].shape[0]

    def tree_flatten(self):
        children = (self.initial_state_logits,
                    self.transition_logits,
                    self.initial_means,
                    self.initial_scale_trils,
                    self.dynamics_matrices,
                    self.dynamics_biases,
                    self.dynamics_scale_trils,
                    self.emissions_matrices,
                    self.emissions_biases,
                    self.emissions_scale_trils)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def initial_distribution(self):
        return tfd.JointDistributionNamed(dict(
            z=          tfd.Categorical(logits=self.initial_state_logits),
            x=lambda z: tfd.MultivariateNormalTriL(loc=self.initial_means[z],
                                                   scale_tril=self.initial_scale_trils[z])
        ))

    def dynamics_distribution(self, state):
        z_prev = state["z"]
        x_prev = state["x"]
        return tfd.JointDistributionNamed(dict(
            z=          tfd.Categorical(logits=self.transition_logits[z_prev]),
            x=lambda z: tfd.MultivariateNormalTriL(
                loc=self.dynamics_matrices[z] @ x_prev + self.dynamics_biases[z],
                scale_tril=self.dynamics_scale_trils[z])
        ))

    def emissions_distribution(self, state):
        z = state["z"]
        x = state["x"]
        return tfd.MultivariateNormalTriL(
            loc=self.emissions_matrices[z] @ x + self.emissions_biases[z],
            scale_tril=self.emissions_scale_trils[z])
