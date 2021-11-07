import jax.numpy as np
import jax.scipy as spsp
from jax import vmap
from jax.tree_util import register_pytree_node_class

import ssm.distributions as ssmd


class Transitions:
    """
    Base class for HMM transitions models,

    .. math::
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self, state):
        """
        Return the conditional distribution of z_t given state z_{t-1}
        """
        raise NotImplementedError

    def log_probs(self, data):
        r"""Returns the log probability of data where

        .. math::
            \texttt{log_P}[i, j] = \log \Pr(z_{t+1} = j | z_t = i)

        if the transition probabilities are stationary or

        .. math::
            \texttt{log_P}[t, i, j] = \log \Pr(z_{t+1} = j | z_t = i)

        if they are nonstationary.

        Args:
            data (np.ndarray): observed data

        Returns:
            log probs (np.ndarray): log probability as defined above

        """
        # TODO: incorporate data or covariates?
        inds = np.arange(self.num_states)
        return vmap(lambda i: self.distribution(i).log_prob(inds))(inds)

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StationaryTransitions(Transitions):
    """
    Basic transition model with a fixed transition matrix.
    """
    def __init__(self,
                 num_states: int,
                 transition_matrix=None,
                 transition_distribution: ssmd.Categorical=None,
                 transition_distribution_prior: ssmd.Dirichlet=None) -> None:
        super(StationaryTransitions, self).__init__(num_states)

        assert transition_matrix is not None or transition_distribution is not None

        if transition_matrix is not None:
            self._distribution = ssmd.Categorical(logits=np.log(transition_matrix))
        else:
            self._distribution = transition_distribution

        if transition_distribution_prior is None:
            num_states = self._distribution.probs_parameter().shape[-1]
            transition_distribution_prior = \
                ssmd.Dirichlet(1.1 * np.ones((num_states, num_states)))
        self._prior = transition_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   transition_distribution=distribution,
                   transition_distribution_prior=prior)

    @property
    def transition_matrix(self):
        return self._distribution.probs_parameter()

    def distribution(self, state):
       return self._distribution[state]

    def transition_log_probs(self, data):
        log_P = self._distribution.logits_parameter()
        log_P -= spsp.logsumexp(log_P, axis=1, keepdims=True)
        return log_P

    def m_step(self, dataset, posteriors):
        stats = np.sum(posteriors.expected_transitions, axis=0)
        stats += self._prior.concentration
        conditional = ssmd.Categorical.compute_conditional_from_stats(stats)
        self._distribution = ssmd.Categorical.from_params(conditional.mode())


@register_pytree_node_class
class StationaryStickyTransitions(Transitions):
    """
    HMM transition model where diagonal elements are a learned
    parameter (alpha) and off-diagonal elements are uniform. That is,
    for a model with n hidden states, the transition matrix is:

        alpha * eye(n) + (1 - alpha) * ones(n, n) / (n - 1)

    where 0 <= alpha <= 1 is a learned parameter.
    """
    def __init__(self,
                 num_states: int,
                 alpha: float=None,
                 transition_distribution: tfd.Categorical=None,
                 transition_distribution_prior: tfd.Beta=None) -> None:

        super(StationaryStickyTransitions, self).__init__(num_states)

        assert transition_distribution is not None or alpha is not None

        # Use specified transition matrix.
        if transition_distribution is not None:
            self.alpha = transition_distribution.probs_parameter()[0, 0]
            self._distribution = transition_distribution

        # If the transition matrix is not given, recompute it given alpha.
        else:
            assert (alpha >= 0) and (alpha <= 1)
            self.alpha = alpha
            self._distribution = tfd.Categorical(
                logits=self._recompute_log_transition_matrix()
            )

        # default prior, expected dwell prob = 0.9
        if transition_distribution_prior is None:
            transition_distribution_prior = tfd.Beta(9, 1)
        self._distribution_prior = transition_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = (self.num_states, self.alpha)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, alpha = aux_data
        distribution, prior = children
        return cls(num_states, alpha,
                   transition_distribution=distribution,
                   transition_distribution_prior=prior)

    def _recompute_log_transition_matrix(self):
        return np.log(
            self.alpha * np.eye(num_states) +
            (1 - self.alpha) * np.ones((num_states, num_states)) / (num_states - 1)
        )

    @property
    def transition_matrix(self):
        return self._distribution.probs_parameter()

    def distribution(self, state):
        return self._distribution[state]

    def m_step(self, dataset, posteriors):

        # Compute num_states x num_states matrix where i, j
        # holds the expected number of transitions from state i
        # to state j.
        stats = np.sum(posteriors.expected_transitions, axis=0)

        # Compute the posterior over alpha, which is a Beta
        # distribution with parameters (c1, c0).
        c1 = (
            np.sum(stats[np.diag_indices_from(stats)]) +
            self._distribution_prior.concentration1
        )
        c1_plus_c0 = (
            np.sum(stats) +
            self._distribution_prior.concentration0
        )

        # Set alpha to the mode of the posterior distribution,
        # which is given by (c1 - 1) / (c1 + c2 - 2) for
        # a Beta distribution.
        self.alpha = (c1 - 1) / (c1_plus_c0 - 2)

        # Recompute the log transition matrix.
        self._distribution = tfd.Categorical(
            logits=self._recompute_log_transition_matrix()
        )

