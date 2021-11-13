import jax.numpy as np
from jax import vmap
from jax.tree_util import tree_map, register_pytree_node_class

import ssm.distributions as ssmd


class Dynamics:
    """
    Base class for HMM transitions models,

    .. math::
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self):
        pass

    def distribution(self, state, covariates=None, metadata=None):
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
                 dynamics_distribution: ssmd.GaussianLinearRegression=None,
                 dynamics_distribution_prior: ssmd.GaussianLinearRegressionPrior=None) -> None:
        super(StationaryDynamics, self).__init__()

        assert (weights is not None and \
                bias is not None and \
                scale_tril is not None) \
            or dynamics_distribution is not None

        if weights is not None:
            self._distribution = ssmd.GaussianLinearRegression(weights, bias, scale_tril)
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
        return cls(aux_data,
                   dynamics_distribution=distribution,
                   dynamics_distribution_prior=prior)

    @property
    def weights(self):
        return self._distribution.weights

    @property
    def bias(self):
        return self._distribution.bias

    @property
    def scale_tril(self):
        return self._distribution.scale_tril

    @property
    def scale(self):
        return self._distribution.scale

    def distribution(self, state, covariates=None, metadata=None):
        if covariates is not None:
            return self._distribution.predict(np.concatenate([state, covariates]))
        else:
            return self._distribution.predict(state)

    def m_step(self,
               batched_data,
               batched_posteriors,
               batched_covariates=None,
               batched_metadata=None):

        # Manually extract the expected sufficient statistics from posterior
        def compute_stats_and_counts(data, posterior, covariates, metadata):
            Ex = posterior.expected_states[:-1]
            Ey = posterior.expected_states[1:]
            ExxT = posterior.expected_states_squared[:-1]
            EyxT = posterior.expected_states_next_states
            EyyT = posterior.expected_states_squared[1:]

            # Concatenate with the covariates
            if covariates is not None:
                u = covariates[1:]
                Ex = np.column_stack((Ex, u))
                ExxT = vmap(lambda xi, xixiT, ui: \
                    np.block([[xixiT,            np.outer(xi, ui)],
                              [np.outer(ui, xi), np.outer(ui, ui)]]))(Ex, ExxT, u)
                EyxT = vmap(lambda yi, yixiT, ui: \
                    np.block([yixiT, np.outer(yi, ui)]))(Ey, EyxT, u)

            # Sum over time
            sum_x = Ex.sum(axis=0)
            sum_y = Ey.sum(axis=0)
            sum_xxT = ExxT.sum(axis=0)
            sum_yxT = EyxT.sum(axis=0)
            sum_yyT = EyyT.sum(axis=0)
            T = len(data) - 1
            stats = (T, sum_xxT, sum_x, T, sum_yxT, sum_y, sum_yyT)
            return stats

        stats = vmap(compute_stats_and_counts)(batched_data,
                                               batched_posteriors,
                                               batched_covariates,
                                               batched_metadata)
        stats = tree_map(sum, stats)  # sum out batch for each leaf

        if self._prior is not None:
            stats = tree_map(np.add, stats, self._prior.natural_parameters)

        conditional = ssmd.GaussianLinearRegression.compute_conditional_from_stats(stats)
        self._distribution = ssmd.GaussianLinearRegression.from_params(conditional.mode())
