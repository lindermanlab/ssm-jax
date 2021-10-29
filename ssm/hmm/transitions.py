import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
from jax.tree_util import register_pytree_node_class

tfd = tfp.distributions

class Transitions:
    """
    Base class for HMM transitions models,

    ..math:
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def distribution(self, state):
        """
        Return the conditional distribution of z_t given state z_{t-1}
        """
        raise NotImplementedError

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StationaryTransitions(Transitions):
    """
    Basic transition model with a fixed transition matrix.
    """
    def __init__(self,
                 transition_matrix=None,
                 transition_distribution: tfd.Categorical=None,
                 transition_distribution_prior: tfd.Dirichlet=None) -> None:

        assert transition_matrix is not None or transition_distribution is not None

        if transition_matrix is not None:
            self._transition_distribution = tfd.Categorical(logits=np.log(transition_matrix))
        else:
            self._transition_distribution = transition_distribution

        if transition_distribution_prior is None:
            num_states = self._transition_distribution.event_shape_tensor[-1]
            transition_distribution_prior = \
                tfd.Dirichlet(1.1 * np.ones((num_states, num_states)))
        self._transition_distribution_prior = transition_distribution_prior

    def tree_flatten(self):
        children = dict(
            transition_distribution=self._transition_distribution,
            transition_distribution_prior=self._transition_distribution_prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**children)

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs_parameter()

    def distribution(self, state):
       return self._transition_distribution[state]

    def m_step(self, dataset, posteriors):
        stats = np.sum(posteriors.expected_transitions, axis=0)
        stats += self._transition_distribution_prior.concentration
        conditional =  tfp.distributions.Dirichlet(concentration=stats)
        self._transition_distribution = tfp.distributions.Categorical(probs=conditional.mode())

