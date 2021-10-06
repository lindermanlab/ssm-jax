"""
see emission.py for design notes
"""

import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import tensorflow_probability.substrates.jax as tfp

@register_pytree_node_class
class InitialDistribution:
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
class CategoricalInitialDistribution(InitialDistribution):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
        stats, counts = (posterior.expected_states[0],), 1

        if prior is not None:
            # Get stats from the prior
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.initial_prior)
        else:
            # Default to uniform prior (0 stats, 1 counts)
            prior_stats, prior_counts = (np.ones(self.num_states) + 1e-4,), 0

        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())
