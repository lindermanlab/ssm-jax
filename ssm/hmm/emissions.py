from functools import partial
import jax.numpy as np
from jax import vmap, lax
from jax.tree_util import tree_map, register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp
import ssm.distributions.expfam as expfam
import ssm.distributions as ssmd
tfd = tfp.distributions


class Emissions:
    """
    Base class of emission distribution of an HMM

    ..  math::
        p_t(x_t \mid z_t, u_t)

    where u_t are optional covariates.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self, state, covariates=None):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        raise NotImplementedError

    def log_probs(self, data):
        """
        Compute log p(x_t | z_t=k) for all t and k.
        """
        inds = np.arange(self.num_states)
        return vmap(lambda k: self.distribution(k).log_prob(data))(inds).T

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class GaussianEmissions(Emissions):
    def __init__(self,
                 num_states: int,
                 means: np.ndarray=None,
                 covariances: np.ndarray=None,
                 emissions_distribution: tfd.MultivariateNormalLinearOperator=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Gaussian Emissions for HMM.
        
        Can be initialized by specifying parameters or by passing in a pre-initialized 
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (np.ndarray, optional): state-dependent emission means. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (tfd.MultivariateNormalLinearOperator, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """
        
        super(GaussianEmissions, self).__init__(num_states)

        assert (means is not None and covariances is not None) \
            or emissions_distribution is not None

        if means is not None and covariances is not None:
            self._distribution = \
                tfd.MultivariateNormalTriL(means, np.linalg.cholesky(covariances))
        else:
            self._distribution = emissions_distribution

        self.distribution_prior = emissions_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self.distribution_prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    def distribution(self, state: int, covariates: np.ndarray=None) -> tfd.MultivariateNormalLinearOperator:
        """Get the distribution at the provided state.

        Args:
            state (int): discrete state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        return self._distribution[state]

    def m_step(self, dataset, posteriors):
        """Update the emissions distribution in-place using an M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors ([type]): the HMM posteriors
        """
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_dataset = flatten(dataset)
        flat_weights = flatten(posteriors.expected_states)

        stats = vmap(expfam._mvn_suff_stats)(flat_dataset)
        stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        counts = flat_weights.sum(axis=0)

        # Add the prior
        if self.distribution_prior is not None:
            prior_stats, prior_counts = expfam._niw_pseudo_obs_and_counts(self.distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the posterior
        conditional = expfam._niw_from_stats(stats, counts)
        mean, covariance = conditional.mode()

        # Set the emissions to the posterior mode
        self._distribution = \
            tfp.distributions.MultivariateNormalTriL(mean, np.linalg.cholesky(covariance))


@register_pytree_node_class
class PoissonEmissions(Emissions):
    def __init__(self,
                 num_states: int,
                 rates: np.ndarray=None,
                 emissions_distribution: tfd.Distribution=None,
                 emissions_distribution_prior: tfd.Gamma=None) -> None:
        """Poisson Emissions for HMM.
        
        Can be initialized by specifying parameters or by passing in a pre-initialized 
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            rates (np.ndarray, optional): state-dependent Poisson rates. Defaults to None.
            emissions_distribution (tfd.Distribution, optional): pre-initialized emissions distribution. 
                Defaults to None.
            emissions_distribution_prior (tfd.Gamma, optional): pre-initialized emissions distribution prior.
                Defaults to None.
        """
        super(PoissonEmissions, self).__init__(num_states)

        assert rates is not None or emissions_distribution is not None

        if rates is not None:
            self._distribution = \
                tfd.Independent(tfd.Poisson(rates), reinterpreted_batch_ndims=1)
        else:
            self._distribution = emissions_distribution

        self._distribution_prior = emissions_distribution_prior

    def distribution(self, state, covariates=None):
        """Get the distribution at the provided state.

        Args:
            state (int): discrete state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        return self._distribution[state]

    def m_step(self, dataset, posteriors):
        """Update the emissions distribution in-place using an M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors ([type]): the HMM posteriors
        """
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_dataset = flatten(dataset)
        flat_weights = flatten(posteriors.expected_states)

        stats = vmap(expfam._poisson_suff_stats)(flat_dataset)
        stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        # counts: (num_states, 1) to broadcast across multiple emission dims
        counts = flat_weights.sum(axis=0)[:, None]

        # Add the prior
        if self._distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam._gamma_pseudo_obs_and_counts(self._distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the posterior
        conditional = expfam._gamma_from_stats(stats, counts)

        # Set the emissions to the posterior mode
        self._distribution = \
            tfp.distributions.Independent(
                tfp.distributions.Poisson(conditional.mode()),
                reinterpreted_batch_ndims=1)

    def tree_flatten(self):
        children = (self._distribution, self._distribution_prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)
