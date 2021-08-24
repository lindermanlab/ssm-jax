import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

from ssm.models.base import SSM


@register_pytree_node_class
class GaussianLDS(SSM):
    def __init__(self,
                 initial_mean,
                 initial_scale_tril,
                 dynamics_matrix,
                 dynamics_bias,
                 dynamics_scale_tril,
                 emissions_matrix,
                 emissions_bias,
                 emissions_scale_tril):
        """ TODO
        """
        # TODO: parameter checking
        self._initial_mean = initial_mean
        self._initial_scale_tril = initial_scale_tril
        self._dynamics_matrix = dynamics_matrix
        self._dynamics_bias = dynamics_bias
        self._dynamics_scale_tril = dynamics_scale_tril
        self._emissions_matrix = emissions_matrix
        self._emissions_bias = emissions_bias
        self._emissions_scale_tril = emissions_scale_tril

    @property
    def latent_dim(self):
        return self._initial_mean.shape[0]

    @property
    def emissions_dim(self):
        return self._emissions_bias.shape[0]

    def tree_flatten(self):
        children = (self._initial_mean,
                    self._initial_scale_tril,
                    self._dynamics_matrix,
                    self._dynamics_bias,
                    self._dynamics_scale_tril,
                    self._emissions_matrix,
                    self._emissions_bias,
                    self._emissions_scale_tril)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        initial_mean, initial_scale_tril, \
            dynamics_matrix, dynamics_bias, dynamics_scale_tril, \
                emissions_matrix, emissions_bias, emissions_scale_tril = children

        return cls(initial_mean,
                   initial_scale_tril,
                   dynamics_matrix,
                   dynamics_bias,
                   dynamics_scale_tril,
                   emissions_matrix,
                   emissions_bias,
                   emissions_scale_tril)

    def initial_dist(self):
        return tfp.distributions.MultivariateNormalTriL(self._initial_mean, self._initial_scale_tril)

    def dynamics_dist(self, state):
        return tfp.distributions.MultivariateNormalTriL(
            self._dynamics_matrix @ state + self._dynamics_bias,
            self._dynamics_scale_tril
        )

    def emissions_dist(self, state):
        return tfp.distributions.MultivariateNormalTriL(
            self._emissions_matrix @ state + self._emissions_bias,
            self._emissions_scale_tril
        )

    def natural_parameters(self, data):
        """ TODO
        """
        seq_len = data.shape[0]

        # Shorthand names for parameters
        m1, chol_Q1 = self._initial_mean, self._initial_scale_tril
        A, b, chol_Q = self._dynamics_matrix, self._dynamics_bias, self._dynamics_scale_tril
        C, d, chol_R = self._emissions_matrix, self._emissions_bias, self._emissions_scale_tril

        # TODO: Work with cholesky factors directly using scipy.linalg.cho_solve
        Q1 = chol_Q1 @ chol_Q1.T
        Q = chol_Q @ chol_Q.T
        R = chol_R @ chol_R.T

        # diagonal blocks of precision matrix
        J_diag = np.dot(C.T, np.linalg.solve(R, C))
        J_diag = np.tile(J_diag[None, :, :], (seq_len, 1, 1))
        J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        h = np.dot(data - d, np.linalg.solve(R, C))
        h.at[0].add(np.linalg.solve(Q1, m1))
        h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h.at[1:].add(np.linalg.solve(Q, b))

        logc = -0.5 * seq_len * self.latent_dim * np.log(2 * np.pi)
        logc += -0.5 * np.linalg.slogdet(Q1)[1]
        logc += -0.5 * np.dot(m1, np.linalg.solve(Q1, m1))
        logc += -0.5 * (seq_len - 1) * np.linalg.slogdet(Q)[1]
        logc += -0.5 * (seq_len - 1) * np.dot(b, np.linalg.solve(Q, b))
        logc += -0.5 * seq_len * self.emissions_dim * np.log(2 * np.pi)
        logc += -0.5 * seq_len * np.linalg.slogdet(R)[1]
        logc += -0.5 * np.sum((y-d) * np.linalg.solve(R, (data - d).T).T)

        return J_diag, J_lower_diag, h, logc
