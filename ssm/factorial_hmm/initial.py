from __future__ import annotations
import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from flax.core.frozen_dict import freeze, FrozenDict

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.hmm.initial import InitialCondition, StandardInitialCondition


@register_pytree_node_class
class FactorialInitialCondition(InitialCondition):
    """
    Initial distribution for several indpendent discrete states.
    """
    def __init__(self,
                 initial_probs: (tuple or list)=None,
                 initial_condition_objects: (tuple or list)=None) -> None:

        if initial_probs is not None:
            initial_condition_objects = \
                tuple(StandardInitialCondition(len(probs), probs) for probs in initial_probs)
        else:
            assert initial_condition_objects is not None, \
                "Must specify either `initial_probs` or `initial_condition_objects`"

        self._initial_conditions = initial_condition_objects
        self.num_groups = len(initial_condition_objects)
        num_states = tuple(ic.num_states for ic in initial_condition_objects)
        super(FactorialInitialCondition, self).__init__(num_states)

    def tree_flatten(self):
        aux_data = None
        children = self._initial_conditions
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(initial_condition_objects=children)
    
    @property
    def _parameters(self):
        return freeze({idx: ic._parameters for idx, ic in enumerate(self._initial_conditions)})
        
    @_parameters.setter
    def _parameters(self, params):
        for idx in range(self.num_groups):
            self._initial_conditions[idx]._parameters = params[idx]
        
    @property
    def _hyperparameters(self):
        return freeze({idx: ic._hyperparameters for idx, ic in enumerate(self._initial_conditions)})
    
    @_hyperparameters.setter
    def _hyperparameters(self, hyperparams):
        for idx in range(self.num_groups):
            self._initial_conditions[idx]._hyperparameters = hyperparams[idx]

    def log_initial_probs(self, data, covariates=None, metadata=None):
        return tuple(ic.log_initial_probs(data, covariates=covariates, metadata=metadata)
                     for ic in self._initial_conditions)

    def distribution(self, covariates=None, metadata=None):
        Root = tfd.JointDistributionCoroutine.Root
        def model():
            for ic in self._initial_conditions:
                yield Root(ic.distribution())
        return tfd.JointDistributionCoroutine(model)

    def m_step(self, data, posterior, covariates=None, metadata=None) -> FactorialInitialCondition:

        class DummyPosterior:
            def __init__(self, expected_initial_states) -> None:
                self.expected_initial_states = expected_initial_states

        for ic, expected_initial_states in \
            zip(self._initial_conditions, posterior.expected_initial_states):
            ic.m_step(data, DummyPosterior(expected_initial_states))

        return self
