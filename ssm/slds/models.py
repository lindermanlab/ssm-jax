import jax.numpy as np
import jax.nn as nn
from jax.tree_util import register_pytree_node_class
import inspect

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.base import SSM
import ssm.hmm.initial
import ssm.hmm.transitions
from ssm.utils import Verbosity, random_rotation, make_named_tuple, ensure_has_batch_dim, auto_batch


# class SLDS(SSM):
#     def __init__(self,
#                  discrete_state_initial_distribution: ssm.hmm.initial.InitialCondition,
#                  continuous_state_initial_distribution: ssm.lds.initial.InitialCondition,
#                  discrete_state_transitions: ssm.hmm.transitions.Transitions,
#                  continuous_state_dynamics: ssm.lds.dynamics.Dynamics,
#                  emissions_distribution: ssm.lds.emissions.Emissions,
#                  ) -> None:
#         super().__init__()

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

        # Grab the parameter values.  This allows us to explicitly re-build the object.
        self._parameters = make_named_tuple(dict_in=locals(),
                                            keys=list(inspect.signature(self.__init__)._parameters.keys()),
                                            name=str(self.__class__.__name__) + 'Tuple')

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
            discrete=tfd.Categorical(logits=self.initial_state_logits),
            continuous=lambda discrete: \
                tfd.MultivariateNormalTriL(loc=self.initial_means[discrete],
                                           scale_tril=self.initial_scale_trils[discrete])
        ))

    def dynamics_distribution(self, state):
        z_prev = state["discrete"]
        x_prev = state["continuous"]
        return tfd.JointDistributionNamed(dict(
            discrete=tfd.Categorical(logits=self.transition_logits[z_prev]),
            continuous=lambda discrete: tfd.MultivariateNormalTriL(
                loc=self.dynamics_matrices[discrete] @ x_prev + self.dynamics_biases[discrete],
                scale_tril=self.dynamics_scale_trils[discrete])
        ))

    def emissions_distribution(self, state):
        z = state["discrete"]
        x = state["continuous"]
        return tfd.MultivariateNormalTriL(
            loc=self.emissions_matrices[z] @ x + self.emissions_biases[z],
            scale_tril=self.emissions_scale_trils[z])
