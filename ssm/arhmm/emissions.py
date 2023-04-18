from __future__ import annotations
from functools import partial
import jax.numpy as np
from jax import vmap, lax
from jax.tree_util import tree_map, register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

from ssm.hmm.emissions import Emissions
from ssm.hmm.posterior import StationaryHMMPosterior
import ssm.distributions as ssmd
tfd = tfp.distributions


@register_pytree_node_class
class AutoregressiveEmissions(Emissions):
    def __init__(self,
                 num_states: int,
                 weights: np.ndarray=None,
                 biases: np.ndarray=None,
                 covariances: np.ndarray=None,
                 emissions_distribution: ssmd.GaussianLinearRegression=None,
                 emissions_distribution_prior: ssmd.GaussianLinearRegressionPrior=None) -> None:
        r"""Gaussian linear regression emissions class for Autoregressive HMM.

        Can be instantiated by specifying the parameters or you can pass in
        the initialized distribution object directly to ``emissions_distribution``.

        Optionally takes an emissions prior distribution.

        Args:
            num_states (int): number of discrete states
            weights (np.ndarray, optional): state-based weight matrix for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim} * \text{num\_lags})`.
                Defaults to None.
            biases (np.ndarray, optional): state-based bias vector for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim})`.
                Defaults to None.
            covariances (np.ndarray, optional): state-based covariances for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Defaults to None.
            emissions_distribution (ssmd.GaussianLinearRegression, optional): initialized emissions distribution. Defaults to None.
            emissions_distribution_prior (ssmd.MatrixNormalInverseWishart, optional): emissions prior distribution. Defaults to None.
        """

        super(AutoregressiveEmissions, self).__init__(num_states)

        params_given = None not in (weights, biases, covariances)
        assert params_given or emissions_distribution is not None

        if params_given:
            self._distribution = \
                ssmd.GaussianLinearRegression(weights, biases, np.linalg.cholesky(covariances))
        else:
            self._distribution = emissions_distribution

        # if emissions_distribution_prior is None:
        #     out_dim = self._distribution.data_dimension
        #     in_dim = self._distribution.covariate_dimensin + 1
        #     self._distribution_prior = \
        #         ssmd.MatrixNormalInverseWishart(
        #             loc=np.zeros((out_dim, in_dim)),
        #             column_covariance=1e8 * np.eye(in_dim),
        #             df=0,
        #             scale=
        #         )
        # else:
        #     self._distribution_prior = emissions_distribution_prior
        self._prior = emissions_distribution_prior

    @property
    def emissions_shape(self):
        return (self._distribution.weights.shape[-1],)

    def distribution(self, state: int, covariates=None, metadata=None, history: np.ndarray=None) -> ssmd.GaussianLinearRegression:
        """Returns the emissions distribution conditioned on a given state.

        Args:
            state (int): latent state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions_distribution (ssmd.GaussianLinearRegression): the emissions distribution
        """
        return self._distribution[state].predict(history.ravel())

    def log_likelihoods_scan(self, data):
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

    def log_likelihoods(self, data, covariates=None, metadata=None):
        # Constants
        num_timesteps, dim = data.shape
        num_states = self.num_states
        num_lags = self._distribution.covariate_dimension // dim

        # Parameters
        weights = self._distribution.weights
        biases = self._distribution.bias
        scale_trils = self._distribution.scale_tril

        # Compute the predictive mean using a 2D convolution
        # TODO: Do we have to flip the weights along the lags dimension?
        mean = lax.conv(data.reshape(1, 1, num_timesteps, dim),
                        weights.reshape(num_states * dim, 1, num_lags, dim),
                        window_strides=(1, 1),
                        padding='VALID')
        mean = mean[0].reshape(num_states, dim, num_timesteps - num_lags + 1).transpose([2, 0, 1])
        # The means are shifted by one so that mean[t] is really the mean of data[t+1].
        mean = mean[:-1] + biases

        # Compute the log probs. Ignore likelihood of the first bit of
        # data since we don't have a prefix
        log_probs = tfd.MultivariateNormalTriL(mean, scale_trils).log_prob(data[num_lags:, None, :])
        log_probs = np.row_stack([np.zeros((num_lags, num_states)), log_probs])
        return log_probs

    def m_step(self,
               dataset: np.ndarray,
               posteriors: StationaryHMMPosterior,
               covariates=None,
               metadata=None) -> AutoregressiveEmissions:
        r"""Update the distribution with an M step.

        Operates over a batch of data.

        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{batch\_dim}, \text{num\_timesteps}, \text{emissions\_dim})`.
            posteriors (StationaryHMMPosterior): HMM posterior object
                with batch_dim to match dataset.
                
        Returns:
            emissions (AutoregressiveEmissions): updated emissions object
        """
        # TODO: Can we compute the expected sufficient statistics with a convolution or scan?

        # weights are shape (num_states, dim, dim * lag)
        num_states = self._distribution.weights.shape[0]
        dim = self._distribution.weights.shape[1]
        num_lags = self._distribution.weights.shape[2] // dim

        # Collect statistics with a scan over data
        def _collect_stats(carry, args):
            x, stats = carry
            y, w = args

            new_x = np.row_stack([x[1:], y])
            new_stats = \
                tree_map(np.add, stats,
                         tree_map(lambda s: np.einsum('k,...->k...', w, s),
                                  ssmd.GaussianLinearRegression.sufficient_statistics(y, x.ravel())))
            return (new_x, new_stats), None

        # Initialize the stats to zero
        init_stats = (np.zeros(num_states),
                      np.zeros((num_states, num_lags * dim, num_lags * dim)),
                      np.zeros((num_states, num_lags * dim)),
                      np.zeros(num_states),
                      np.zeros((num_states, dim, num_lags * dim)),
                      np.zeros((num_states, dim)),
                      np.zeros((num_states, dim, dim)))

        # Scan over one time series
        def scan_one(data, weights):
            (_, stats), _ = lax.scan(_collect_stats,
                                     (data[:num_lags], init_stats),
                                     (data[num_lags:], weights[num_lags:]))
            return stats

        # vmap over all time series in dataset
        print(dataset[0].shape)
        print(posteriors[0].expected_states.shape)
        stats = [vmap(scan_one)(dataset[i], posteriors[i].expected_states) for i in range(len(dataset))]
        print(stats[0].shape)
        stats = np.concatenate(stats)
        stats = tree_map(partial(np.sum, axis=0), stats)

        # Add the prior stats and counts
        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        # Compute the conditional distribution over parameters and take the mode
        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())
        return self

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)
