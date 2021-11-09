import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.hmm.transitions import Transitions


@register_pytree_node_class
class FactorialStationaryTransitions(Transitions):
    """
    Transitions over several hidden states evolving in parallel.
    """
    def __init__(self, transitions) -> None:

        self.num_groups = len(transitions)
        self._transitions = transitions
        num_states = np.prod([t.num_states for t in transitions])
        super(FactorialStationaryTransitions, self).__init__(num_states)

    def tree_flatten(self):
        # children, aux_data = [], []
        # for t in self._transitions:
        #     c, a = t.tree_flatten()
        #     children.append(c)
        #     aux_data.append((type(t), *a))
        # return tuple(children), tuple(aux_data)
        aux_data = None
        children = self._transitions
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # transitions = []
        # for aux, chd in zip(aux_data, children):
        #     sub_cls = aux[0]
        #     transitions.append(sub_cls.tree_unflatten(aux[1:], chd))
        # return cls(transitions)
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

    def m_step(self, dataset, posteriors):

        class DummyPosterior:
            def __init__(self, expected_transitions) -> None:
                self.expected_transitions = expected_transitions

        for transitions_object, expected_transitions in \
            zip(self._transitions, posteriors.expected_transitions):
            transitions_object.m_step(dataset, DummyPosterior(expected_transitions))

        # # Sum over trials.
        # stats = np.sum(expected_transitions, axis=0)

        # # Iterate over groups of latent variables.
        # for i, t in enumerate(self._transitions):

        #     # Sum over axes associated with other transitions.
        #     ax = [j for j in range(stats.ndim)]
        #     ax.remove(i)
        #     ax.remove(i + self.num_groups)
        #     reduced_stats = np.sum(stats, axis=ax)

        #     # Perform m-step for transition operator t.
        #     t.m_step(dataset, reduced_stats)
