import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map, register_pytree_node_class

import ssm.distributions as ssmd


class Dynamics:
    """
    Base class for SLDS dynamics model

    .. math::
        p_t(x_t \mid x_{t-1}, z_t)
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
class StandardDynamics(Dynamics):
    """
    Basic dynamics model for LDS.
    """
    def __init__(self,
                 weights=None,
                 biases=None,
                 scale_trils=None,
                 dynamics_distribution: ssmd.GaussianLinearRegression=None,
                 dynamics_distribution_prior: ssmd.GaussianLinearRegressionPrior=None) -> None:
        super(StandardDynamics, self).__init__()

        assert (weights is not None and \
                biases is not None and \
                scale_trils is not None) \
            or dynamics_distribution is not None

        if weights is not None:
            self._distribution = ssmd.GaussianLinearRegression(weights, biases, scale_trils)
        else:
            self._distribution = dynamics_distribution

        if dynamics_distribution_prior is None:
            pass  # TODO: implement default prior
        self._prior = dynamics_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(dynamics_distribution=distribution,
                   dynamics_distribution_prior=prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def biases(self):
        return self._distribution.bias

    @property
    def scale_trils(self):
        return self._distribution.scale_tril

    @property
    def scales(self):
        return self._distribution.scale

    def distribution(self, prev_continuous_state, discrete_state, covariates=None, metadata=None):
        d = self._distribution[discrete_state]
        if covariates is not None:
            return d.predict(np.concatenate([prev_continuous_state, covariates]))
        else:
            return d.predict(prev_continuous_state)

    def m_step(self, dataset, posteriors):
        # TODO: Compute expected sufficient statistics under q(z).

        # Manually extract the expected sufficient statistics from posterior
        def compute_stats_and_counts(data, posterior):
            Ez = posterior.discrete_posterior.expected_states
            Ex = posterior.continuous_posterior.expected_states
            ExxT = posterior.continuous_posterior.expected_states_squared
            ExnxT = posterior.continuous_posterior.expected_states_next_states

            # Sum over time
            sum_x = np.einsum('tk,ti->ki', Ez[1:], Ex[:-1])
            sum_y = np.einsum('tk,ti->ki', Ez[1:], Ex[1:])
            sum_xxT = np.einsum('tk,tij->kij', Ez[1:], ExxT[:-1])
            sum_yxT = np.einsum('tk,tij->kij', Ez[1:], ExnxT)
            sum_yyT = np.einsum('tk,tij->kij', Ez[1:], ExxT[1:])
            T = np.sum(Ez[1:], axis=0)
            stats = (T, sum_xxT, sum_x, T, sum_yxT, sum_y, sum_yyT)
            return stats

        stats = vmap(compute_stats_and_counts)(dataset, posteriors)
        stats = tree_map(sum, stats)  # sum out batch for each leaf

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())
