import jax.numpy as np
import jax.scipy.special as spsp
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

class InitialCondition:
    """
    Base class for initial state distributions of an HMM.

    .. math::
        p(z_1 \mid u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self):
        """
        Return the distribution of z_1
        """
        raise NotImplementedError

    def log_probs(self, data):
        """
        Return [log Pr(z_1 = k) for k in range(num_states)]
        """
        return self.distribution().log_prob(np.arange(self.num_states))

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StandardInitialCondition(InitialCondition):
    """
    The standard model is a categorical distribution.
    """
    def __init__(self,
                 num_states: int,
                 initial_probs=None,
                 initial_distribution: tfd.Categorical=None,
                 initial_distribution_prior: tfd.Dirichlet=None) -> None:
        super(StandardInitialCondition, self).__init__(num_states)

        assert initial_probs is not None or initial_distribution is not None

        if initial_probs is not None:
            self._distribution = tfd.Categorical(logits=np.log(initial_probs))
        else:
            self._distribution = initial_distribution
        num_states = self._distribution.probs_parameter().shape[-1]

        if initial_distribution_prior is None:
            initial_distribution_prior = tfd.Dirichlet(1.1 * np.ones(num_states))
        self._distribution_prior = initial_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   initial_distribution=distribution,
                   initial_distribution_prior=prior)

    def distribution(self):
       return self._distribution

    def initial_log_probs(self, data):
        """
        Return [log Pr(z_1 = k) for k in range(num_states)]
        """
        lps = self._distribution.logits_parameter()
        return lps - spsp.logsumexp(lps)

    def m_step(self, dataset, posteriors):
        stats = np.sum(posteriors.expected_states[:, 0, :], axis=0)
        stats += self._distribution_prior.concentration
        conditional = tfp.distributions.Dirichlet(concentration=stats)
        self._distribution = tfp.distributions.Categorical(probs=conditional.mode())
