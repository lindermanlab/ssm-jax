"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from functools import partial
from typing import Any
Array = Any

import jax.numpy as np
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class, tree_map

from tensorflow_probability.substrates import jax as tfp

import ssm.distributions
import ssm.distributions.expfam as expfam
from ssm.hmm.base import HMM, AutoregressiveHMM


class _GaussianEmissionsMixin(object):
    """An HMM with Gaussian emissions."""

    def _m_step_emission_distribution(self, dataset, posteriors):
        """If we have the right posterior, we can perform an exact update here.
        """
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_dataset = flatten(dataset)
        flat_weights = flatten(posteriors.expected_states)

        stats = vmap(expfam._mvn_suff_stats)(flat_dataset)
        stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        counts = flat_weights.sum(axis=0)

        # Add the prior
        if self._emission_distribution_prior is not None:
            prior_stats, prior_counts = expfam._niw_pseudo_obs_and_counts(self._emission_distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the posterior
        conditional = expfam._niw_from_stats(stats, counts)
        mean, covariance = conditional.mode()

        # Set the emissions to the posterior mode
        self._emission_distribution = \
            tfp.distributions.MultivariateNormalTriL(mean, np.linalg.cholesky(covariance))


@register_pytree_node_class
class GaussianHMM(_GaussianEmissionsMixin, HMM):
    pass


class _PoissonEmissionsMixin(object):
    """
    TODO
    """
    def _m_step_emission_distribution(self, dataset, posteriors):
        flatten = lambda x: x.reshape(-1, x.shape[-1])
        flat_dataset = flatten(dataset)
        flat_weights = flatten(posteriors.expected_states)

        stats = vmap(expfam._poisson_suff_stats)(flat_dataset)
        stats = tree_map(lambda x: np.einsum('nk,n...->k...', flat_weights, x), stats)
        # counts: (num_states, 1) to broadcast across multiple emission dims
        counts = flat_weights.sum(axis=0)[:, None]

        # Add the prior
        if self._emission_distribution_prior is not None:
            prior_stats, prior_counts = \
                expfam._gamma_pseudo_obs_and_counts(self._emission_distribution_prior)
            stats = tree_map(np.add, stats, prior_stats)
            counts = counts + prior_counts

        # Compute the posterior
        conditional = expfam._gamma_from_stats(stats, counts)

        # Set the emissions to the posterior mode
        self._emission_distribution = \
            tfp.distributions.Independent(
                tfp.distributions.Poisson(conditional.mode()),
                reinterpreted_batch_ndims=1)


@register_pytree_node_class
class PoissonHMM(_PoissonEmissionsMixin, HMM):
    pass


class _GaussianAutoregressiveMixin(object):

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


@register_pytree_node_class
class GaussianAutoregressiveHMM(_GaussianAutoregressiveMixin, AutoregressiveHMM):
    pass
