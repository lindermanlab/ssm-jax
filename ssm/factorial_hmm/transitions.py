import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.hmm.transitions import Transitions


@register_pytree_node_class
class FactorialTransitions(Transitions):
    """
    Transitions over several hidden states evolving in parallel.
    """
    def __init__(self, transitions) -> None:

        num_states = tuple(t.num_states for t in transitions)
        self.num_groups = len(transitions)
        self._transitions = transitions
        super(FactorialTransitions, self).__init__(num_states)

    def tree_flatten(self):
        aux_data = None
        children = self._transitions
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    @property
    def transition_matrix(self):
        M = np.ones(1)
        for t in self._transitions:
            M = np.kron(M, t.transition_matrix)
        return M

    def distribution(self, state):
        Root = tfd.JointDistributionCoroutine.Root
        def model():
            for prev_state, transitions in zip(state, self._transitions):
                yield Root(transitions.distribution(prev_state))
        return tfd.JointDistributionCoroutine(model)

    def log_probs(self, data):
        return tuple(t.log_probs(data) for t in self._transitions)

    def m_step(self, dataset, posteriors):

        class DummyPosterior:
            def __init__(self, expected_transitions) -> None:
                self.expected_transitions = expected_transitions

        for transitions_object, expected_transitions in \
            zip(self._transitions, posteriors.expected_transitions):
            transitions_object.m_step(dataset, DummyPosterior(expected_transitions))

        return self
