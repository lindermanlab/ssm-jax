import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.hmm.initial import InitialCondition

@register_pytree_node_class
class FactorialInitialCondition(InitialCondition):
    """
    Initial distribution for several indpendent discrete states.
    """
    def __init__(self, initial_conditions) -> None:

        self.num_groups = len(initial_conditions)
        self._initial_conditions = initial_conditions
        num_states = (ic.num_states for ic in initial_conditions)
        super(FactorialInitialCondition, self).__init__(num_states)

    def tree_flatten(self):
        aux_data = None
        children = self._initial_conditions
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def log_probs(self, data):
        lp = 0
        for g, ic in enumerate(self._initial_conditions):
            lp_expanded = np.expand_dims(ic.log_probs(data), axis=range(1, self.num_groups))
            lp += np.swapaxes(lp_expanded, 0, g)
        return lp

    def distribution(self):
        Root = tfd.JointDistributionCoroutine.Root
        def model():
            for ic in self._initial_conditions:
                yield Root(ic.distribution())
        return tfd.JointDistributionCoroutine(model)

    def m_step(self, dataset, posteriors):

        class DummyPosterior:
            def __init__(self, expected_states) -> None:
                self.expected_states = expected_states

        for g, ic in enumerate(self._initial_conditions):
            # Marginalize over all but this group
            # (first two axes are batches and time steps)
            axes = np.concatenate([np.arange(g), np.arange(g+1, self.num_groups)]) + 2
            expected_states = np.sum(posteriors.expected_states, axis=axes)
            assert expected_states.ndim == 3 and np.allclose(expected_states.sum(axis=2), 1.0)
            ic.m_step(dataset, DummyPosterior(expected_states))
