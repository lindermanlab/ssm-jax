from functools import partial
import jax.numpy as np
from jax import vmap, lax
from jax.tree_util import tree_map, register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

from ssm.factorial_hmm.emissions import FactorialEmissions
from ssm.hmm.posterior import StationaryHMMPosterior
import ssm.distributions as ssmd
tfd = tfp.distributions


@register_pytree_node_class
class TimeWarpedAutoregressiveEmissions(FactorialEmissions):
    def __init__(self,
                 num_discrete_states: int,
                 time_constants: np.ndarray,
                 weights: np.ndarray=None,
                 biases: np.ndarray=None,
                 covariances: np.ndarray=None) -> None:
                #  emissions_distribution: ssmd.GaussianLinearRegression=None,
                #  emissions_distribution_prior: ssmd.GaussianLinearRegressionPrior=None
        r"""Gaussian linear regression emissions class for Autoregressive HMM.

        Can be instantiated by specifying the parameters or you can pass in
        the initialized distribution object directly to ``emissions_distribution``.

        Optionally takes an emissions prior distribution.

        ..math:
            x_t \sim N((I + \frac{1}{\tau_t} A_{z_t}) x_{t-1} + \frac{1}{\tau_t} b_{z_t}, Q_{z_t})

        Args:
            num_states (int): number of discrete states
            time_constants (np.ndarray): array of non-negative time constants
            weights (np.ndarray, optional): state-based weight matrix for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim} * \text{num\_lags})`.
                Note: dynamics are I + 1/\tau A where A are the weights.
                Defaults to None.
            biases (np.ndarray, optional): state-based bias vector for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim})`.
                Note: dynamics are 1/\tau b where b are the biases.
                Defaults to None.
            covariances (np.ndarray, optional): state-based covariances for Gaussian linear regression
                of shape :math:`(\text{num\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Note: covariances are 1/\tau Q where Q are the covariance matrices.
                Defaults to None.
            emissions_distribution (ssmd.GaussianLinearRegression, optional): initialized emissions distribution. Defaults to None.
            emissions_distribution_prior (ssmd.MatrixNormalInverseWishart, optional): emissions prior distribution. Defaults to None.
        """
        self.num_discrete_states = num_discrete_states
        self.time_constants = time_constants
        num_time_constants = len(time_constants)
        num_states = (num_discrete_states, num_time_constants)
        self._weights = weights
        self._biases = biases
        self._covariances = covariances
        super(TimeWarpedAutoregressiveEmissions, self).__init__(num_states)

        # Make the distribution object
        latent_dim = weights.shape[-1]
        effective_weights = np.einsum('kde,i->kide', weights, 1/time_constants) + np.eye(latent_dim)
        effective_biases = np.einsum('kd,i->kid', biases, 1/time_constants)
        effective_covariances = np.einsum('kde,i->kide', covariances, 1/time_constants)
        self._distribution = \
            ssmd.GaussianLinearRegression(effective_weights,
                                            effective_biases,
                                            np.linalg.cholesky(effective_covariances))
        # self._prior = emissions_distribution_prior

    @property
    def emissions_dim(self):
        return self._distribution.weights.shape[-1]

    def distribution(self, state: int, covariates: np.ndarray=None) -> ssmd.GaussianLinearRegression:
        """Returns the emissions distribution conditioned on a given state.

        Args:
            state (int): latent state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions_distribution (ssmd.GaussianLinearRegression): the emissions distribution
        """
        return self._distribution[state]

    # def log_probs_scan(self, data):
    #     # Compute the emission log probs
    #     dim = self._distribution.data_dimension
    #     num_lags = self._distribution.covariate_dimension // dim

    #     # Scan over the data
    #     def _compute_ll(x, y):
    #         ll = self._distribution.log_prob(y, covariates=x.ravel())
    #         new_x = np.row_stack([x[1:], y])
    #         return new_x, ll
    #     _, log_probs = lax.scan(_compute_ll, np.zeros((num_lags, dim)), data)

    #     # Ignore likelihood of the first bit of data since we don't have a prefix
    #     log_probs = log_probs.at[:num_lags].set(0.0)
    #     return log_probs

    def log_probs(self, data):
        # Constants
        num_timesteps, dim = data.shape
        num_states = self.num_states
        num_lags = self._distribution.covariate_dimension // dim

        # Parameters
        weights = self._weights
        biases = self._biases
        covariances = self._covariances

        # Compute the predictive mean using a 2D convolution
        # TODO: Do we have to flip the weights along the lags dimension?
        mean = lax.conv(data.reshape(1, 1, num_timesteps, dim),
                        weights.reshape(num_states * dim, 1, num_lags, dim),
                        window_strides=(1, 1),
                        padding='VALID')
        mean = mean[0].reshape(num_states, dim, num_timesteps - num_lags + 1).transpose([2, 0, 1])

        # The means are shifted by one so that mean[t] is really the mean of data[t+1].
        mean = mean[:-1] + biases

        # TODO: scale mean by 1/tau and evaluate log prob of *Delta x* rather than x

        # Compute the log probs. Ignore likelihood of the first bit of
        # data since we don't have a prefix
        log_probs = tfd.MultivariateNormalFullCovariance(mean, covariances).log_prob(data[num_lags:, None, :])
        log_probs = np.row_stack([np.zeros((num_lags, num_states)), log_probs])
        return log_probs

    def m_step(self, dataset: np.ndarray, posteriors: StationaryHMMPosterior) -> None:
        r"""Update the distribution (in-place) with an M step.

        Operates over a batch of data.

        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{batch\_dim}, \text{num\_timesteps}, \text{emissions\_dim})`.
            posteriors (StationaryHMMPosterior): HMM posterior object
                with batch_dim to match dataset.
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
        stats = vmap(scan_one)(dataset, posteriors.expected_states)
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
        aux_data = self.num_discrete_states, self.time_constants
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)
