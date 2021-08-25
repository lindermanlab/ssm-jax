import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

from ssm.models.base import SSM


@register_pytree_node_class
class GaussianLDS(SSM):
    def __init__(self,
                 initial_distribution,
                 dynamics_distribution,
                 emissions_distribution):
        """ TODO
        """
        # TODO: parameter checking
        self._initial_distribution = initial_distribution
        self._dynamics_distribution = dynamics_distribution
        self._emissions_distribution = emissions_distribution

    @property
    def latent_dim(self):
        return self._emissions_distribution.weights.shape[-1]

    @property
    def emissions_dim(self):
        return self._emissions_distribution.weights.shape[-2]

    def tree_flatten(self):
        children = (self._initial_distribution,
                    self._dynamics_distribution,
                    self._emissions_distribution)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        initial_distribution, dynamics_distribution, emissions_distribution = children

        return cls(initial_distribution,
                   dynamics_distribution,
                   emissions_distribution)

    def initial_distribution(self):
        return self._initial_distribution

    def dynamics_distribution(self, state):
        return self._dynamics_distribution.predict(state)

    def emissions_distribution(self, state):
        return self._emissions_distribution.predict(state)

    def natural_parameters(self, data):
        """ TODO
        """
        seq_len = data.shape[0]

        # Shorthand names for parameters
        m1 = self._initial_distribution.loc
        chol_Q1 = self._initial_distribution.scale_tril
        A = self._dynamics_distribution.weights
        b = self._dynamics_distribution.bias
        chol_Q = self._dynamics_distribution.scale_tril
        C = self._emissions_distribution.weights
        d = self._emissions_distribution.bias
        chol_R = self._emissions_distribution.scale_tril

        # TODO: Work with cholesky factors directly using scipy.linalg.cho_solve
        Q1 = chol_Q1 @ chol_Q1.T
        Q = chol_Q @ chol_Q.T
        R = chol_R @ chol_R.T

        # diagonal blocks of precision matrix
        J_diag = np.dot(C.T, np.linalg.solve(R, C))
        J_diag = np.tile(J_diag[None, :, :], (seq_len, 1, 1))
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag = J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag = J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        h = np.dot(data - d, np.linalg.solve(R, C))
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h = h.at[1:].add(np.linalg.solve(Q, b))

        logc = -0.5 * seq_len * self.latent_dim * np.log(2 * np.pi)
        logc += -0.5 * np.linalg.slogdet(Q1)[1]
        logc += -0.5 * np.dot(m1, np.linalg.solve(Q1, m1))
        logc += -0.5 * (seq_len - 1) * np.linalg.slogdet(Q)[1]
        logc += -0.5 * (seq_len - 1) * np.dot(b, np.linalg.solve(Q, b))
        logc += -0.5 * seq_len * self.emissions_dim * np.log(2 * np.pi)
        logc += -0.5 * seq_len * np.linalg.slogdet(R)[1]
        logc += -0.5 * np.sum((data - d) * np.linalg.solve(R, (data - d).T).T)

        return J_diag, J_lower_diag, h, logc
