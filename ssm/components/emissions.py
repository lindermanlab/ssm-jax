"""
emissions.py
===============

PROTOTYPE:

General design notes:
=====================

- Emissions object should lighlty wrap TFP distributions to provide friendly interface upstream
    - interface should include:
        - .permute(permutation)                        ==> permute the latent states defined by the distribution
        - .exact_m_step(data, posterior, prior=None)   ==> (if possible) return an updated distribution
        - .sgd_update(data, posterior)                 ==> SGD on -log_prob objective (when m_step isn't available)
        - .initialize(data)                            ==> Initialize parameters of distribution in data-aware way
        - [...]
- Like the ssm.jax_refactor branch, we could bundle a list of different distributions / state

Issues:
=======
    - right now, there is redundancy for the ExpFam distribution M_steps.
        - can we quarter that off somewhere else?
    - is this abstraction layer necessary?
        - will try to clean up other stuff to see if this makes sense
    - how should approximate m steps (i.e. Laplace) be handled? 
        - can the details of the algorithm be abstracted away?
        - does it make sense for the inference details to be implemented here?
"""
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import tensorflow_probability.substrates.jax as tfp

@register_pytree_node_class
class Emissions:
    def __init__(self, num_states, distribution):
        assert isinstance(distribution, tfp.distributions.Distribution)
        self.distribution = distribution
        self.num_states = num_states

    def exact_m_step(self, data, posterior, prior=None):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.distribution)
        aux_data = (self.num_states,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, = aux_data
        distribution = children
        return cls(num_states,
                   distribution=distribution)


@register_pytree_node_class
class GaussianEmissions(Emissions):
    def exact_m_step(self, data, posterior, prior=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        expfam = EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"]
        stats = tuple(
            np.einsum('tk,t...->k...', posterior.expected_states, s)
            for s in expfam.suff_stats(data))
        counts = np.sum(posterior.expected_states, axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())

@register_pytree_node_class
class PoissonEmissions(Emissions):
    def exact_m_step(self, data, posterior, prior=None):
        """If we have the right posterior, we can perform an exact update here.
        """
        expfam = EXPFAM_DISTRIBUTIONS["IndependentPoisson"]
        stats = tuple(
            np.einsum('tk,t...->k...', posterior.expected_states, s)
            for s in expfam.suff_stats(data))
        counts = np.sum(posterior.expected_states, axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())