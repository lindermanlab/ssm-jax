"""
LDS Initial Distribution Classes
================================
"""
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, tree_util, vmap
from jax.tree_util import register_pytree_node_class
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.models.base.components import ContinuousComponent
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


@register_pytree_node_class
class GaussianInitialDistribution(ContinuousComponent):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["MultivariateNormalTriL"]

        def compute_stats_and_counts(data, posteiror):
            Ex = posterior.mean[0]
            ExxT = posterior.expected_states_squared[0]
            stats = (1.0, Ex, ExxT)
            counts = 1.0
            return stats, counts

        stats, counts = vmap(compute_stats_and_counts)(data, posterior)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.initial_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())
