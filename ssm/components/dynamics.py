"""
see emissions.py for design notes
"""


import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import tensorflow_probability.substrates.jax as tfp

@register_pytree_node_class
class Dynamics:
    def __init__(self, num_states, distribution):
        assert isinstance(distribution, tfp.distributions.Distribution)
        self.distribution = distribution
        self.num_states = num_states

    def exact_m_step(self, data, posterior, prior=None):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.distribution,)
        aux_data = (self.num_states,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, = aux_data
        distribution, = children
        return cls(num_states,
                   distribution=distribution)

# TODO: expfam updates should be elsewhere (perhaps a core inference module)
# Then, we could call the appropriate update here
@register_pytree_node_class
class CategoricalDynamics(Dynamics):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
        stats, counts = (posterior.expected_transitions,), 0

        if prior is not None:
            # Get stats from the prior
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.transition_prior)
        else:
            # Default to uniform prior (0 stats, 1 counts)
            prior_stats, prior_counts = (np.ones((self.num_states, self.num_states)) + 1e-4,), 0

        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())


@register_pytree_node_class
class GaussianLinearRegressionDynamics(Dynamics):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["GaussianGLM"]

        # Extract expected sufficient statistics from posterior
        Ex = posterior.mean
        ExxT, ExnxT = posterior.second_moments

        # Sum over time
        sum_x = Ex[:-1].sum(axis=0)
        sum_y = Ex[1:].sum(axis=0)
        sum_xxT = ExxT[:-1].sum(axis=0)
        sum_yxT = ExnxT.sum(axis=0)
        sum_yyT = ExxT[1:].sum(axis=0)
        stats = (sum_x, sum_y, sum_xxT, sum_yxT, sum_yyT)
        counts = len(data) - 1

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.dynamics_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())