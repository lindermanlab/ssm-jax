"""
LDS Dynamics Classes
====================

* GaussianLinearRegressionDynamics

"""
import jax.numpy as np
from jax import tree_util, vmap
from jax.tree_util import register_pytree_node_class
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from ssm.distributions.expfam import EXPFAM_DISTRIBUTIONS
from ssm.distributions import GaussianLinearRegression
from ssm.utils import sum_tuples


class Dynamics:
    """
    Base class for HMM transitions models,

    ..math:
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self):
        pass

    def distribution(self, state):
        """
        Return the conditional distribution of z_t given state z_{t-1}
        """
        raise NotImplementedError

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StationaryDynamics(Dynamics):
    """
    Basic dynamics model for LDS.
    """
    def __init__(self,
                 weights=None,
                 bias=None,
                 scale_tril=None,
                 dynamics_distribution: GaussianLinearRegression=None,
                 dynamics_distribution_prior: tfd.Distribution=None) -> None:
        super(StationaryDynamics, self).__init__()

        assert (weights is not None and \
                bias is not None and \
                scale_tril is not None) \
            or dynamics_distribution is not None

        if weights is not None:
            self._dynamics_distribution = GaussianLinearRegression(weights, bias, scale_tril)
        else:
            self._dynamics_distribution = dynamics_distribution

        if dynamics_distribution_prior is None:
            pass  # TODO: implement default prior
        self._dynamics_distribution_prior = dynamics_distribution_prior

    def tree_flatten(self):
        children = (self._dynamics_distribution, self._dynamics_distribution_prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   dynamics_distribution=distribution,
                   dynamics_distribution_prior=prior)

    @property
    def weights(self):
        return self._dynamics_distribution.weights

    @property
    def bias(self):
        return self._dynamics_distribution.bias

    @property
    def scale_tril(self):
        return self._dynamics_distribution.scale_tril

    @property
    def scale(self):
        return self._dynamics_distribution.scale

    def distribution(self, state):
       return self._dynamics_distribution.predict(covariates=state)

    def m_step(self, dataset, posteriors):

        expfam = EXPFAM_DISTRIBUTIONS["GaussianLinearRegression"]

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

        stats, counts = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_util.tree_map(sum, stats)  # sum out batch for each leaf
        counts = counts.sum(axis=0)

        if self._dynamics_distribution_prior  is not None:
            prior_stats, prior_counts = \
                expfam.prior_pseudo_obs_and_counts(self._dynamics_distribution_prior )
            stats = sum_tuples(stats, prior_stats)
            counts += prior_counts

        param_posterior = expfam.posterior_from_stats(stats, counts)
        self._dynamics_distribution = expfam.from_params(param_posterior.mode())
