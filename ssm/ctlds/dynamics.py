import jax.numpy as np
from jax.scipy.linalg import expm
from jax import vmap
from jax.tree_util import tree_map, register_pytree_node_class

import ssm.distributions as ssmd


class Dynamics:
    """
    Base class for CT-LDS dynamics models,

    .. math::
        p_t(z_t \mid z_{t-1}, u_t)

    where u_t are optional covariates at time t.
    """
    def __init__(self):
        pass

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of x_t given state x_{t-1}
        
        Args:
            state (float): state x_{t-1}
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
                
        Returns:
            distribution (tfd.Distribution): conditional distribution of x_t given state x_{t-1}.
        """
        raise NotImplementedError

    def m_step(self, dataset, posteriors):
        """Update the transition parameters in an M step given posteriors
        over the latent states. 
        
        Update is performed in place.

        Args:
            dataset (np.ndarray): the observed dataset with shape (B, T, D)
            posteriors (HMMPosterior): posteriors over the latent states with leaf shape (B, ...)
            covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
                Defaults to None.
            metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
                Defaults to None.
        """
        # TODO: implement generic m-step
        raise NotImplementedError


@register_pytree_node_class
class StationaryDynamics(Dynamics):
    """
    Basic dynamics model for CTLDS.

    Uses the underlying SDE:
        dx = (drift_matrix @ x + drift_bias) dt + scale @ dB
    where B is a standard Brownian motion.
    """
    def __init__(self,
                 drift_matrix=None,
                 drift_bias=None,
                 scale=None,
                 dynamics_distribution_prior=None) -> None:
        super(StationaryDynamics, self).__init__()

        assert (drift_matrix is not None and \
                drift_bias is not None and \
                scale is not None)

        self.drift_matrix = drift_matrix
        self.drift_bias = drift_bias
        self.scale = scale

        self._state_dim = drift_matrix.shape[0]
    
        if dynamics_distribution_prior is None:
            pass  # TODO: implement default prior
        self._prior = dynamics_distribution_prior

    def tree_flatten(self):
        children = (self.drift_matrix, self.drift_bias, self.scale, self._prior) # will this work?
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        drift_matrix, drift_bias, scale, prior = children
        return cls(aux_data,
                   drift_matrix=drift_matrix,
                   drift_bias=drift_bias,
                   scale=scale,
                   dynamics_distribution_prior=prior)

    def distribution(self, state, covariates=None, metadata=None):
        # Assume state is (B, S), covariates is (B,), metadata is None.

        def compute_transition_density(state, covariates, metadata):
            zeros = np.zeros_like(self.drift_matrix)
            augmented_drift_matrix = (
                np.block([[self.drift_matrix, np.eye(self._state_dim)],
                          [zeros,             zeros]]))
            augmented_transition_matrix = expm(augmented_drift_matrix * covariates)
            transition_matrix = augmented_transition_matrix[:self._state_dim, :self._state_dim]
            bias = augmented_transition_matrix[:self._state_dim, self._state_dim:] @ self.drift_bias

            hamiltonian = (
                np.block([[self.drift_matrix, self.scale @ self.scale.T],
                          [zeros,             -self.drift_matrix.T]]))
            matrix_fraction_numerator = expm(hamiltonian * covariates)[:self._state_dim, self._state_dim:]
            noise_covariance = matrix_fraction_numerator @ transition_matrix.T
            noise_covariance_tril = np.linalg.cholesky(noise_covariance)

            return ssmd.GaussianLinearRegression(transition_matrix, bias, noise_covariance_tril).predict(state)

        return vmap(compute_transition_density)(state, covariates, metadata)

    def m_step(self,
               batched_data,
               batched_posteriors,
               batched_covariates=None,
               batched_metadata=None):

        # TODO: This needs to be changed.

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
