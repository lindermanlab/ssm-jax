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

    ..math:
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
                 means=None,
                 covariances=None,
                 emissions_distribution: tfd.MultivariateNormalLinearOperator=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
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

    def distribution(self, state, covariates=None):
        return self._distribution[state]

    def m_step(self, dataset, posteriors):
        """If we have the right posterior, we can perform an exact update here.
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
    """
    TODO
    """
    def __init__(self,
                 num_states: int,
                 rates=None,
                 emissions_distribution: tfd.Distribution=None,
                 emissions_distribution_prior: tfd.Gamma=None) -> None:
        super(PoissonEmissions, self).__init__(num_states)

        assert rates is not None or emissions_distribution is not None

        if rates is not None:
            self._distribution = \
                tfd.Independent(tfd.Poisson(rates), reinterpreted_batch_ndims=1)
        else:
            self._distribution = emissions_distribution

        self._distribution_prior = emissions_distribution_prior

    def distribution(self, state, covariates=None):
        return self._distribution[state]

    def m_step(self, dataset, posteriors):
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


@register_pytree_node_class
class AutoregressiveEmissions(Emissions):

    def __init__(self,
                 num_states,
                 weights=None,
                 biases=None,
                 covariances=None,
                 emissions_distribution: ssmd.GaussianLinearRegression=None,
                 emissions_distribution_prior: ssmd.MatrixNormalInverseWishart=None) -> None:
        super(AutoregressiveEmissions, self).__init__(num_states)

        params_given = None not in (weights, biases, covariances)
        assert params_given or emissions_distribution is not None

        if params_given:
            self._distribution = \
                ssmd.GaussianLinearRegression(weights, biases, np.linalg.cholesky(covariances))
        else:
            self._distribution = emissions_distribution

        self._distribution_prior = emissions_distribution_prior

    def distribution(self, state, covariates=None):
        return self._distribution[state]

    def log_probs(self, data):
        # Compute the emission log probs
        dim = self._distribution.data_dimension
        num_lags = self._distribution.covariate_dimension // dim

        # Scan over the data
        def _compute_ll(x, y):
            ll = self._distribution.log_prob(y, covariates=x.ravel())
            new_x = np.row_stack([x[1:], y])
            return new_x, ll
        _, log_probs = lax.scan(_compute_ll, np.zeros((num_lags, dim)), data)

        # Ignore likelihood of the first bit of data since we don't have a prefix
        log_probs = log_probs.at[:num_lags].set(0.0)
        return log_probs

    def m_step(self, dataset, posteriors):
        """
        Can we compute the expected sufficient statistics with a convolution or scan?
        """
        # weights are shape (num_states, dim, dim * lag)
        num_states = self._distribution.weights.shape[0]
        dim = self._distribution.weights.shape[1]
        num_lags = self._distribution.weights.shape[2] // dim

        # Collect statistics with a scan over data
        def _collect_stats(carry, args):
            x, stats, counts = carry
            y, w = args

            new_x = np.row_stack([x[1:], y])
            new_stats = tree_map(np.add, stats,
                                 tree_map(lambda s: np.einsum('k,...->k...', w, s),
                                          expfam._gaussian_linreg_suff_stats(y, x.ravel())))
            new_counts = counts + w
            return (new_x, new_stats, new_counts), None

        # Initialize the stats and counts to zero
        init_stats = (np.zeros((num_states, num_lags * dim)),
                      np.zeros((num_states, dim)),
                      np.zeros((num_states, num_lags * dim, num_lags * dim)),
                      np.zeros((num_states, dim, num_lags * dim)),
                      np.zeros((num_states, dim, dim)))
        init_counts = np.zeros(num_states)

        # Scan over one time series
        def scan_one(data, weights):
            (_, stats, counts), _ = lax.scan(_collect_stats,
                                             (data[:num_lags], init_stats, init_counts),
                                             (data[num_lags:], weights[num_lags:]))
            return stats, counts

        # vmap over all time series in dataset
        stats, counts = vmap(scan_one)(dataset, posteriors.expected_states)
        stats = tree_map(partial(np.sum, axis=0), stats)
        counts = np.sum(counts, axis=0)

        # Add the prior stats and counts
        if self._distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam._mniw_pseudo_obs_and_counts(self._distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the conditional distribution over parameters
        conditional = expfam._mniw_from_stats(stats, counts)

        # Set the emissions to the posterior mode
        weights_and_bias, covariance_matrix = conditional.mode()
        weights, bias = weights_and_bias[..., :-1], weights_and_bias[..., -1]
        self._distribution = \
            ssmd.GaussianLinearRegression(
                weights, bias, np.linalg.cholesky(covariance_matrix))

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
