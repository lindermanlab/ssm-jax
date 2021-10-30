import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

class InitialCondition:
    """
    Base class for initial state distributions of an HMM.

    ..math:
        p(z_1 \mid u_t)

    where u_t are optional covariates at time t.
    """
    def distribution(self):
        """
        Return the distribution of z_1
        """
        raise NotImplementedError

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StandardInitialCondition(InitialCondition):
    """
    The standard model is a categorical distribution.
    """
    def __init__(self,
                 initial_probs=None,
                 initial_distribution: tfd.Categorical=None,
                 initial_distribution_prior: tfd.Dirichlet=None) -> None:

        assert initial_probs is not None or initial_distribution is not None

        if initial_probs is not None:
            self._initial_distribution = tfd.Categorical(logits=np.log(initial_probs))
        else:
            self._initial_distribution = initial_distribution
        num_states = self._initial_distribution.probs_parameter().shape[-1]

        if initial_distribution_prior is None:
            initial_distribution_prior = tfd.Dirichlet(1.1 * np.ones(num_states))
        self._initial_distribution_prior = initial_distribution_prior

    def tree_flatten(self):
        children = (self._initial_distribution, self._initial_distribution_prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(initial_distribution=distribution,
                   initial_distribution_prior=prior)

    def distribution(self):
       return self._initial_distribution

    def m_step(self, dataset, posteriors):
        stats = np.sum(posteriors.expected_states[:, 0, :], axis=0)
        stats += self._initial_distribution_prior.concentration
        conditional = tfp.distributions.Dirichlet(concentration=stats)
        self._initial_distribution = tfp.distributions.Categorical(probs=conditional.mode())
