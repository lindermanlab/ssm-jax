"""
LDS Dynamics Classes
====================
"""
import jax.numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, tree_util, vmap
from jax.tree_util import register_pytree_node_class
from ssm.distributions.expfam import EXPFAM_DISTRIBUTIONS
from ssm.lds.components import ContinuousComponent
from ssm.utils import Verbosity, ssm_pbar, sum_tuples


@register_pytree_node_class
class GaussianLinearRegressionDynamics(ContinuousComponent):
    def exact_m_step(self, data, posterior, prior=None):
        expfam = EXPFAM_DISTRIBUTIONS["GaussianGLM"]

        # Extract expected sufficient statistics from posterior
        def compute_stats_and_counts(data, posterior):
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
            return stats, counts

        stats, counts = vmap(compute_stats_and_counts)(data, posterior)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if prior is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(prior.dynamics_prior)
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        return expfam.from_params(param_posterior.mode())
