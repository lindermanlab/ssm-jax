from functools import partial
import jax.numpy as np
from jax.tree_util import tree_map, register_pytree_node_class
from jax import vmap

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
from ssm.distributions.linreg import GaussianLinearRegression, GaussianLinearRegressionPrior
from ssm.factorial_hmm.emissions import FactorialEmissions
from ssm.factorial_hmm.posterior import FactorialHMMPosterior


@register_pytree_node_class
class TimeWarpedAutoregressiveEmissions(FactorialEmissions):
    def __init__(self,
                 num_discrete_states: int,
                 time_constants: np.ndarray,
                 weights: np.ndarray=None,
                 biases: np.ndarray=None,
                 scale_trils: np.ndarray=None,
                 emissions_distribution_prior: GaussianLinearRegressionPrior=None) -> None:
        r"""Emissions class for a Time-Warped Autoregressive HMM (TWAR-HMM).

        Note: We (currently) only support lag-1 AR-HMMs.

        Optionally takes in a prior distribution.

        ..math:
            x_t \sim N((I + \frac{1}{\tau_t} A_{z_t}) x_{t-1} + \frac{1}{\tau_t} b_{z_t}, \frac{1}{\tau_t} Q_{z_t})

        Args:
            num_states (int): number of discrete states
            time_constants (np.ndarray): array of non-negative time constants
            weights (np.ndarray, optional): state-based weight matrix for Gaussian linear regression
                of shape :math:`(\text{num\_discrete\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Note: dynamics are I + 1/\tau A where A are the weights.
                Defaults to None.
            biases (np.ndarray, optional): state-based bias vector for Gaussian linear regression
                of shape :math:`(\text{num\_discrete\_states}, \text{emissions\_dim})`.
                Note: dynamics are 1/\tau b where b are the biases.
                Defaults to None.
            scale_trils (np.ndarray, optional): state-based scale_trils for Gaussian linear regression
                of shape :math:`(\text{num\_discrete\_states}, \text{emissions\_dim}, \text{emissions\_dim})`.
                Note: scale_trils are 1/\tau chol(Q) where Q are the covariance matrices.
                Defaults to None.
            emissions_distribution_prior (ssmd.MatrixNormalInverseWishart, optional):
                emissions prior distribution. Defaults to None.
        """
        self.num_discrete_states = num_discrete_states
        self._time_constants = time_constants
        num_time_constants = len(time_constants)
        num_states = (num_discrete_states, num_time_constants)
        self._weights = weights
        self._biases = biases
        self._scale_trils = scale_trils
        self._prior = emissions_distribution_prior
        self._make_distribution()
        super(TimeWarpedAutoregressiveEmissions, self).__init__(num_states)

    def _make_distribution(self):
        dim = self._weights.shape[-1]
        effective_weights = np.einsum('kde,i->kide', self._weights, 1/self._time_constants) + np.eye(dim)
        effective_biases = np.einsum('kd,i->kid', self._biases, 1/self._time_constants)
        effective_scale_trils = np.einsum('kde,i->kide', self._scale_trils, 1/np.sqrt(self._time_constants))
        self._distribution = GaussianLinearRegression(
            effective_weights, effective_biases, effective_scale_trils)

    @property
    def emissions_shape(self):
        return (self._weights.shape[-1],)

    @property
    def time_constants(self):
        return self._time_constants

    def distribution(self, state: int, covariates: np.ndarray=None, metadata=None, history: np.ndarray=None) -> GaussianLinearRegression:
        """Returns the emissions distribution conditioned on a given state.

        Args:
            state (int): latent state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions_distribution (GaussianLinearRegression): the emissions distribution
        """
        return self._distribution[state].predict(history.ravel())

    def log_likelihoods(self, data, covariates=None, metadata=None):

        # Warp the covariances
        # warped_scale_trils = np.einsum('kij,c->kcij', self._scale_trils, 1/self.time_constants)
        warped_scale_trils = np.einsum('kij,c->kcij', self._scale_trils, 1/np.sqrt(self.time_constants))

        # For AR(1) models, the (unwarped) predictive change in x is just a matrix multiplication
        def _lp_single(dx, x):
            # Compute the unwarped expected change in x
            means = np.einsum('kij,j->ki', self._weights, x) + self._biases
            # Warp by the time constants
            warped_means = np.einsum('ki,c->kci', means, 1/self.time_constants)
            return tfd.MultivariateNormalTriL(warped_means, warped_scale_trils).log_prob(dx)

        # Evaluate the log prob at each data point
        log_probs = vmap(_lp_single)(np.diff(data, axis=0), data[:-1])
        log_probs = np.concatenate([np.zeros((1,) + self.num_states), log_probs], axis=0)
        return log_probs

    def m_step(self, dataset: np.ndarray,
               posteriors: FactorialHMMPosterior,
               covariates=None,
               metadata=None) -> None:
        r"""Update the distribution (in-place) with an M step.

        The parameters are (A_k, b_k, Q_k) for each of the discrete states. The key idea
        is that once we fix the time-warping constant \tau, the likelihood is just a
        Gaussian linear regression where

        ..math:
            dx_t \sim \mathcal{N}(\frac{1}{\tau_t}(A_{z_t} x_t + b_{z_t}), \frac{1}{\tau_t} Q_{z_t})

        and :math:`dx_t = x_{t+1} - x_t`.

        To update the parameters of the linear regression, we need the expected sufficient statistics,
        which we obtain by expanding the log likelihood above and writing it as a sum of inner
        products between parameters (A, b, Q) and sufficient statistics, which are functions of \tau,
        x, and dx.

        ..math:
            (1,
             E[x_t x_t^\top / \tau_t],
             E[x_t / \tau_t],
             E[1 / \tau_t],
             E[dx_t x_t^\top],
             E[dx_t],
             E[\tau_t dx_t dx_t^\top])

        These are easy to compute because :math:`\tau_t` is the only random variable and it is constrained
        to a finite set of values.

        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{batch\_dim}, \text{num\_timesteps}, \text{emissions\_dim})`.
            posteriors (StationaryHMMPosterior): HMM posterior object
                with batch_dim to match dataset.
        """
        # Collect statistics for a single time step
        def _suff_stats_single(expected_states, dx, x):
            # compute sufficient statistics for each time constant
            # f = lambda tau: GaussianLinearRegression.sufficient_statistics(tau * dx, x)
            # stats = vmap(f)(self.time_constants)
            tw_sufficient_stats = lambda tau: \
                (1.0,
                 1 / tau * np.outer(x, x),
                 1 / tau * x,
                 1 / tau,
                 np.outer(dx, x),
                 dx,
                 tau * np.outer(dx, dx))
            stats = vmap(tw_sufficient_stats)(self.time_constants)

            # compute the expected sufficient statistics by summing over time constants
            # weighted by Pr(z=k, tau=c) for each discrete state k
            return tree_map(lambda s: np.einsum('kc,c...->k...', expected_states, s), stats)

        # Collect the sum of statistics for a single trial
        def _sum_stats_single(data, posterior):
            stats = vmap(_suff_stats_single)(posterior.expected_states[1:],
                                             np.diff(data, axis=0), data[:-1])
            return tree_map(partial(np.sum, axis=0), stats)

        # Sum over trials
        stats = tree_map(partial(np.sum, axis=0), vmap(_sum_stats_single)(dataset, posteriors))

        # Compute the conditional distribution over parameters and take the mode
        conditional = GaussianLinearRegression.compute_conditional_from_stats(stats)
        linreg = GaussianLinearRegression.from_params(conditional.mode())
        self._weights = linreg.weights
        self._biases = linreg.bias
        self._scale_trils = linreg.scale_tril
        self._make_distribution()
        return self

    def tree_flatten(self):
        aux_data = self.num_discrete_states
        children = self.time_constants, self._weights, self._biases, self._scale_trils, self._prior
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)
