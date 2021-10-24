"""
Autoregressive Hidden Markov Model (ARHMM)
==========================================

"""
from functools import partial
from typing import Any
Array = Any

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class, tree_map

import ssm.distributions
import ssm.distributions.expfam as expfam
from ssm.hmm import HMM


@register_pytree_node_class
class AutoregressiveHMM(HMM):
    """
    TODO
    """
    @property
    def emission_dim(self):
        return self._emission_distribution.data_dimension

    @property
    def num_lags(self):
        return self._emission_distribution.covariate_dimension // self._emission_distribution.data_dimension

    def sample(self, key, num_steps: int, initial_state=None, num_samples=1, prev_emissions=None):
        """
        Sample from the joint distribution defined by the state space model.

        .. math::
            x, y \sim p(x, y)

        Args:
            key (PRNGKey): A JAX pseudorandom number generator key.
            num_steps (int): Number of steps for which to sample.
            initial_state: Optional state on which to condition the sampled trajectory.
                Default is None which samples the intial state from the initial distribution.
            prev_emissions: Optional initial emissions to start the autoregressive model.

        Returns:
            states: A ``(timesteps,)`` array of the state value across time (:math:`x`).
            emissions: A ``(timesteps, obs_dim)`` array of the observations across time (:math:`y`).

        """

        def _sample(key):
            if initial_state is None:
                key1, key = jr.split(key, 2)
                state = self.initial_distribution().sample(seed=key1)
            else:
                state = initial_state

            if prev_emissions is None:
                history = np.zeros((self.num_lags, self.emission_dim))
            else:
                history = prev_emissions

            def _step(carry, key):
                history, state = carry
                key1, key2 = jr.split(key, 2)
                emission = self.emissions_distribution(state).sample(seed=key1, covariates=history.ravel())
                next_state = self.dynamics_distribution(state).sample(seed=key2)
                next_history = np.row_stack([history[1:], emission])
                return (next_history, next_state), (state, emission)

            keys = jr.split(key, num_steps)
            _, (states, emissions) = lax.scan(_step, (history, state), keys)
            return states, emissions

        if num_samples > 1:
            batch_keys = jr.split(key, num_samples)
            states, emissions = vmap(_sample)(batch_keys)
        else:
            states, emissions = _sample(key)

        return states, emissions

    def natural_parameters(self, data: Array):
        """Obtain the natural parameters for the HMM given observation data.

        The natural parameters for an HMM are:
            - log probability of the initial state distribution
            - log probablity of the transitions (log transition matrix)
            - log likelihoods of the emissions data

        Args:
            data (Array): Observed data array: ``(time, obs_dim)``.

        Returns:
            log_initial_state_distn (Array): log probability of the initial state distribution
            log_transition_matrix (Array): log of transition matrix
            log_likelihoods (Array): log probability of emissions
        """
        log_initial_state_distn = self._initial_distribution.logits_parameter()
        log_transition_matrix = self._transition_distribution.logits_parameter()
        log_transition_matrix -= spsp.logsumexp(log_transition_matrix, axis=1, keepdims=True)

        def _compute_ll(x, y):
            ll = self._emission_distribution.log_prob(y, covariates=x.ravel())
            new_x = np.row_stack([x[1:], y])
            return new_x, ll

        _, log_likelihoods = lax.scan(_compute_ll, np.zeros((self.num_lags, self.emission_dim)), data)

        # Ignore likelihood of the first bit of data since we don't have a prefix
        log_likelihoods = log_likelihoods.at[:self.num_lags].set(0.0)

        return log_initial_state_distn, log_transition_matrix, log_likelihoods

    def _m_step_emission_distribution(self, dataset, posteriors):
        """
        Can we compute the expected sufficient statistics with a convolution or scan?
        """
        num_states = self.num_states
        num_lags = self.num_lags
        dim = dataset.shape[-1]

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
        if self._emission_distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam._mniw_pseudo_obs_and_counts(self._emission_distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the conditional distribution over parameters
        conditional = expfam._mniw_from_stats(stats, counts)

        # Set the emissions to the posterior mode
        weights_and_bias, covariance_matrix = conditional.mode()
        weights, bias = weights_and_bias[..., :-1], weights_and_bias[..., -1]
        self._emission_distribution = \
            ssm.distributions.GaussianLinearRegression(
                weights, bias, np.linalg.cholesky(covariance_matrix))
