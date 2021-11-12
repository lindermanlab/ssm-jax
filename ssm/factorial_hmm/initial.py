import jax.numpy as np
from jax.tree_util import register_pytree_node_class

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

        self.num_groups = len(initial_condition_objects)
        if initial_probs is not None:
            initial_condition_objects = \
                tuple(StandardInitialCondition(probs) for probs in initial_probs)
        else:
            assert initial_condition_objects is not None, \
                "Must specify either `initial_probs` or `initial_condition_objects`"
        self._initial_conditions = initial_condition_objects
        num_states = tuple(ic.num_states for ic in initial_condition_objects)
        super(FactorialInitialCondition, self).__init__(num_states)

    def tree_flatten(self):
        aux_data = None
        children = self._initial_conditions
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def log_probs(self, data):
        return tuple(ic.log_probs(data) for ic in self._initial_conditions)

    def distribution(self):
        Root = tfd.JointDistributionCoroutine.Root
        def model():
            for ic in self._initial_conditions:
                yield Root(ic.distribution())
        return tfd.JointDistributionCoroutine(model)

    def m_step(self, dataset, posteriors):

        class DummyPosterior:
            def __init__(self, expected_initial_states) -> None:
                self.expected_initial_states = expected_initial_states

        for ic, expected_initial_states in \
            zip(self._initial_conditions, posteriors.expected_initial_states):
            ic.m_step(dataset, DummyPosterior(expected_initial_states))

        return self
