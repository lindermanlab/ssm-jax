from jax._src.tree_util import tree_map
import jax.numpy as np
from jax import tree_util, vmap
from jax.tree_util import register_pytree_node_class

import ssm.distributions as ssmd

class InitialCondition:
    """
    Base class for initial state distributions of an LDS.

    .. math::
        p(x_1 \mid u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self):
        pass

    def distribution(self, covariates=None, metadata=None):
        """
        Return the distribution of x_1 (potentially given covariates u_t)
        
        Args: 
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            distribution (tfd.Distribution): distribution of z_1
        """
        raise NotImplementedError

    def m_step(self, dataset, posteriors, covariates=None, metadata=None):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StandardInitialCondition(InitialCondition):
    """
    The standard model is a multivariate Normal distribution.
    (With covariance parameterized by the lower triagular scale
    cov = scale_tril @ scale_tril.T)
    """
    def __init__(self,
        initial_mean=None,
        initial_scale_tril=None,
        initial_distribution: ssmd.MultivariateNormalTriL=None,
        initial_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        super(StandardInitialCondition, self).__init__()

        assert (initial_mean is not None and initial_scale_tril is not None) or initial_distribution is not None

        if initial_mean is not None:
            self._distribution = ssmd.MultivariateNormalTriL(loc=initial_mean, scale_tril=initial_scale_tril)
        else:
            self._distribution = initial_distribution

        if initial_distribution_prior is None:
            pass  # TODO: implement default prior
        self._prior = initial_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(initial_distribution=distribution,
                   initial_distribution_prior=prior)

    @property
    def mean(self):
        return self._distribution.loc

    def distribution(self, covariates=None, metadata=None):
        """
        Return the distribution of x_1.
        
        Args: 
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            distribution (tfd.Distribution): distribution of z_1
        """
        return self._distribution

    def m_step(self, dataset, posteriors, covariates=None, metadata=None):
        """Update the initial distribution in an M step given posteriors over the latent states. 
        
        Update is performed in place.

        Args:
            dataset (np.ndarray): the observed dataset with shape (B, T, D)
            posteriors (HMMPosterior): posteriors over the latent states with leaf shape (B, ...)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
        """
        def compute_stats_and_counts(data, posterior):
            Ex = posterior.expected_states[0]
            ExxT = posterior.expected_states_squared[0]
            stats = (1.0, Ex, ExxT, 1.0)
            return stats

        stats = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.MultivariateNormalTriL.compute_conditional_from_stats(stats)
        self._distribution = ssmd.MultivariateNormalTriL.from_params(conditional.mode())
