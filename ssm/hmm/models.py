"""
HMM Model Classes
=================

Module defining model behavior for Hidden Markov Models (HMMs).
"""
from functools import partial
from typing import Any
Array = Any

import jax.numpy as np
import jax.scipy.special as spsp
from jax import vmap, lax
from jax.tree_util import register_pytree_node_class, tree_map

from tensorflow_probability.substrates import jax as tfp

import ssm.distributions
import ssm.distributions.expfam as expfam
from ssm.base import SSM
from ssm.inference.em import em
from ssm.hmm.posterior import hmm_expected_states, HMMPosterior
from ssm.utils import Verbosity, format_dataset


class HMM(SSM):

    def __init__(self, num_states: int,
                 initial_distribution: tfp.distributions.Categorical,
                 transition_distribution: tfp.distributions.Categorical,
                 emission_distribution: tfp.distributions.Distribution,
                 initial_distribution_prior: tfp.distributions.Dirichlet=None,
                 transition_distribution_prior: tfp.distributions.Dirichlet=None,
                 emission_distribution_prior: tfp.distributions.Distribution=None,
                 ):
        """Class for Hidden Markov Model (HMM).

        Args:
            num_states (int): Number of discrete latent states.
            initial_distribution (tfp.distributions.Categorical): The distribution over the initial state.
            transition_distribution (tfp.distributions.Categorical): The transition distribution.
        """
        self.num_states = num_states
        self._initial_distribution = initial_distribution
        self._transition_distribution = transition_distribution
        self._emission_distribution = emission_distribution

        # Initialize uniform priors unless otherwise specified
        if initial_distribution_prior is None:
            initial_distribution_prior = \
                tfp.distributions.Dirichlet(1.1 * np.ones(num_states))
        self._initial_distribution_prior = initial_distribution_prior

        if transition_distribution_prior is None:
            transition_distribution_prior = \
                tfp.distributions.Dirichlet(1.1 * np.ones((num_states, num_states)))
        self._transition_distribution_prior = transition_distribution_prior

        # Subclasses can initialize in their constructors this as necessary
        self._emission_distribution_prior = emission_distribution_prior

    def tree_flatten(self):
        children = (self._initial_distribution,
                    self._transition_distribution,
                    self._emission_distribution,
                    self._initial_distribution_prior,
                    self._transition_distribution_prior,
                    self._emission_distribution_prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data, *children)

    def initial_distribution(self):
        return self._initial_distribution

    def dynamics_distribution(self, state):
        return self._transition_distribution[state]

    def emissions_distribution(self, state):
        return self._emission_distribution[state]

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs_parameter()

    ### Methods for posterior inference
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
        log_likelihoods = vmap(lambda k:
                               vmap(lambda x: self.emissions_distribution(k).log_prob(x))(data)
                               )(np.arange(self.num_states)).T

        return log_initial_state_distn, log_transition_matrix, log_likelihoods

    @format_dataset
    def infer_posterior(self, dataset):
        marginal_likelihood, (Ez0, Ezzp1, Ez) = vmap(
            lambda data: hmm_expected_states(*self.natural_parameters(data)))(dataset)
        return HMMPosterior(marginal_likelihood, Ez, Ezzp1)

    @format_dataset
    def marginal_likelihood(self, dataset, posterior=None):
        if posterior is None:
            posterior = self.infer_posterior(dataset)
        return posterior.marginal_likelihood

    ### EM
    def e_step(self, dataset):
        return self.infer_posterior(dataset)

    def _m_step_initial_distribution(self, posteriors):
        stats = np.sum(posteriors.expected_states[:, 0, :], axis=0)
        stats += self._initial_distribution_prior.concentration
        conditional = tfp.distributions.Dirichlet(concentration=stats)
        self._initial_distribution = tfp.distributions.Categorical(probs=conditional.mode())

    def _m_step_transition_distribution(self, posteriors):
        stats = np.sum(posteriors.expected_transitions, axis=0)
        stats += self._transition_distribution_prior.concentration
        conditional =  tfp.distributions.Dirichlet(concentration=stats)
        self._transition_distribution = tfp.distributions.Categorical(probs=conditional.mode())

    def _m_step_emission_distribution(self, dataset, posteriors):
        # TODO: We could do gradient ascent on the expected log likelihood
        raise NotImplementedError

    def m_step(self, dataset, posteriors):
        self._m_step_initial_distribution(posteriors)
        self._m_step_transition_distribution(posteriors)
        self._m_step_emission_distribution(dataset, posteriors)

    @format_dataset
    def fit(self, dataset, method="em", num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):
        model = self
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        if method == "em":
            log_probs, model, posteriors = em(model, dataset, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return log_probs, model, posteriors


@register_pytree_node_class
class GaussianHMM(HMM):
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
class PoissonHMM(HMM):
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

