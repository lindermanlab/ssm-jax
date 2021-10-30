import jax.numpy as np
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax.tree_util import register_pytree_node_class

from ssm.hmm.base import HMM
from ssm.hmm.initial import StandardInitialCondition
from ssm.hmm.transitions import StationaryTransitions
from ssm.hmm.emissions import GaussianEmissions, PoissonEmissions

import warnings

@register_pytree_node_class
class GaussianHMM(HMM):
    def __init__(self,
                 num_states,
                 num_emission_dims=None,
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
        transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)
        emissions = GaussianEmissions(num_states, means=emission_means, covariances=emission_covariances)
        super(GaussianHMM, self).__init__(num_states,
                                          initial_condition,
                                          transitions,
                                          emissions)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    # directly GaussianHMM using parent (HMM) constructor
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(aux_data, *children)
        return obj


@register_pytree_node_class
class PoissonHMM(HMM):
    def __init__(self,
                 num_states,
                 num_emission_dims=None,
                 initial_state_probs=None,
                 transition_matrix=None,
                 emission_rates=None,
                 seed=None):

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

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    # directly init PoissonHMM using parent (HMM) constructor
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(aux_data, *children)
        return obj
