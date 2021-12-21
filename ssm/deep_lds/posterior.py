import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_leaves, register_pytree_node_class

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import auto_batch

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# No need to register because we are not adding additional data, just methods
class LDSSVAEPosterior(MultivariateNormalBlockTridiag):

    @classmethod
    def initialize(cls, svae, data, covariates=None, metadata=None):
        # Create a dummy object
        T = data.shape[-2]
        D = svae.latent_dim
        precision_diag_blocks = np.zeros((T, D, D))
        precision_lower_diag_blocks = np.zeros((T-1, D, D))
        linear_potential = np.zeros((T, D))
        log_normalizer = 1
        filtered_precisions = np.zeros((T, D, D))
        filtered_linear_potentials = np.zeros((T, D))
        expected_states = np.zeros((T, D))
        expected_states_squared = np.zeros((T, D, D))
        expected_states_next_states = np.zeros((T-1, D, D))

        self = cls(precision_diag_blocks,
                 precision_lower_diag_blocks,
                 linear_potential,
                 log_normalizer,
                 filtered_precisions,
                 filtered_linear_potentials,
                 expected_states,
                 expected_states_squared,
                 expected_states_next_states)

        return self
        
    @property
    def emissions_shape(self):
        return (self.precision_diag_blocks.shape[-1],)
        
    @auto_batch(batched_args=("data", "potential", "covariates", "metadata", "key"))
    def update(self, lds, data, potential, covariates=None, metadata=None, key=None):
        # Shorthand names for parameters
        m1 = lds.initial_mean
        Q1 = lds.initial_covariance
        A = lds.dynamics_matrix
        b = lds.dynamics_bias
        Q = lds.dynamics_noise_covariance
        # From data
        J_obs, h_obs = potential
        seq_len = data.shape[0]
        latent_dim = J_obs.shape[-1]

        # diagonal blocks of precision matrix
        J_diag = J_obs  # from observations
        # Expand the diagonals into full covariance matrices
        J_diag = J_obs[..., None] * np.eye(latent_dim)[None, ...]
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag = J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag = J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        # linear potential
        h = h_obs  # from observations
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h = h.at[1:].add(np.linalg.solve(Q, b))

        return LDSSVAEPosterior.infer(J_diag, J_lower_diag, h)

class DKFPosterior(MultivariateNormalBlockTridiag):

    @classmethod
    def initialize(cls, svae, data, covariates=None, metadata=None):
        # Create a dummy object
        T = data.shape[-2]
        D = svae.latent_dim
        precision_diag_blocks = np.zeros((T, D, D))
        precision_lower_diag_blocks = np.zeros((T-1, D, D))
        linear_potential = np.zeros((T, D))
        log_normalizer = 1
        filtered_precisions = np.zeros((T, D, D))
        filtered_linear_potentials = np.zeros((T, D))
        expected_states = np.zeros((T, D))
        expected_states_squared = np.zeros((T, D, D))
        expected_states_next_states = np.zeros((T-1, D, D))

        self = cls(precision_diag_blocks,
                 precision_lower_diag_blocks,
                 linear_potential,
                 log_normalizer,
                 filtered_precisions,
                 filtered_linear_potentials,
                 expected_states,
                 expected_states_squared,
                 expected_states_next_states)

        return self
        
    @property
    def emissions_shape(self):
        return (self.precision_diag_blocks.shape[-1],)
        
    @auto_batch(batched_args=("data", "potential", "covariates", "metadata", "key"))
    def update(self, lds, data, potential, covariates=None, metadata=None, key=None):
        # From data
        J_obs, h_obs = potential
        seq_len = data.shape[0]
        latent_dim = J_obs.shape[-1]
        # diagonal blocks of precision matrix
        # Expand the diagonals into full covariance matrices
        J_diag = J_obs[..., None] * np.eye(latent_dim)[None, ...]
        # lower diagonal blocks of precision matrix
        J_lower_diag = np.zeros((seq_len-1, latent_dim, latent_dim))
        # linear potential
        h = h_obs  # from observations
        return DKFPosterior.infer(J_diag, J_lower_diag, h)