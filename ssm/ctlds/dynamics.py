import jax
import jax.numpy as np
import jax.scipy.linalg as jspla
import jax.scipy.optimize as jspop
from jax import vmap, lax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, register_pytree_node_class

from functools import partial

import ssm.distributions as ssmd
from ssm.lds.dynamics import Dynamics

def compute_transition_params(drift_matrix, drift_bias, diffusion_scale, covariates):
    zeros = np.zeros_like(drift_matrix)
    state_dim = drift_matrix.shape[0]
    augmented_drift_matrix = (
        np.block([[drift_matrix, np.eye(state_dim)],
                  [zeros,        zeros]]))
    augmented_transition_matrix = jspla.expm(augmented_drift_matrix * covariates)
    transition_matrix = augmented_transition_matrix[:state_dim, :state_dim]
    bias = augmented_transition_matrix[:state_dim, state_dim:] @ drift_bias

    hamiltonian = (
        np.block([[drift_matrix, diffusion_scale @ diffusion_scale.T],
                  [zeros,      -drift_matrix.T]]))
    matrix_fraction_numerator = jspla.expm(hamiltonian * covariates)[:state_dim, state_dim:]
    noise_covariance = matrix_fraction_numerator @ transition_matrix.T
    
    return transition_matrix, bias, noise_covariance

compute_transition_params_sequence = vmap(compute_transition_params, in_axes=(None, None, None, 0))
    
@register_pytree_node_class
class StationaryCTDynamics(Dynamics):
    """
    Basic dynamics model for CTLDS.

    Uses the underlying SDE:
        dx = (drift_matrix @ x + drift_bias) dt + diffusion_scale @ dB
    where B is a standard Brownian motion.
    """
    def __init__(self,
                 drift_matrix=None,
                 drift_bias=None,
                 diffusion_scale=None,
                 dynamics_distribution_prior=None) -> None:
        super(StationaryCTDynamics, self).__init__()

        assert (drift_matrix is not None and \
                drift_bias is not None and \
                diffusion_scale is not None)

        self.drift_matrix = drift_matrix
        self.drift_bias = drift_bias
        self.diffusion_scale = diffusion_scale

        self._state_dim = drift_matrix.shape[0]
    
        if dynamics_distribution_prior is None:
            pass  # TODO: implement default prior
        self._prior = dynamics_distribution_prior

    def tree_flatten(self):
        children = (self.drift_matrix, self.drift_bias, self.diffusion_scale, self._prior) # will this work?
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        drift_matrix, drift_bias, diffusion_scale, prior = children
        return cls(aux_data,
                   drift_matrix=drift_matrix,
                   drift_bias=drift_bias,
                   diffusion_scale=diffusion_scale,
                   dynamics_distribution_prior=prior)


    def transition_params(self, covariates):
        return compute_transition_params(self.drift_matrix, self.drift_bias, self.diffusion_scale, covariates)

    def distribution(self, state, covariates=None, metadata=None):
        transition_matrix, bias, noise_covariance = self.transition_params(covariates)
        noise_covariance_tril = np.linalg.cholesky(noise_covariance)

        return ssmd.GaussianLinearRegression(transition_matrix, bias, noise_covariance_tril).predict(state)

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

            stats = (ExxT, Ex, EyxT, Ey, EyyT)
            return stats

        batched_stats = vmap(compute_stats_and_counts)(batched_data,
                                                       batched_posteriors,
                                                       batched_covariates,
                                                       batched_metadata)
        
        flattened_params, unravel = ravel_pytree((self.drift_matrix, self.drift_bias, self.diffusion_scale))

        def _objective(flattened_params):
            drift_matrix, drift_bias, diffusion_scale = unravel(flattened_params) 
            
            def _single_element_objective(stats, covariates):
                ''' stats should be ((T, D, D), (T, D), (T, D, D), (T, D), (T, D, D))
                covariates should be (T,)
                '''                
                # we only ever use linalg.inv(noise_covariance). We can find this directly by
                # taking expm(-hamiltonian) (probably)
                # TODO: should I scale the objective value by (1./T)?
                
                def _one_step_objective(stats, params):
                    ''' stats should be ((D, D), (D,), (D, D), (D,), (D, D))
                    params should be ((D, D), (D,), (D, D))
                    '''
                    A, b, Q = params
                    xxT, x, yxT, y, yyT = stats
                    
                    Q_inv = np.linalg.inv(Q)
                    vol_term = -0.5 * np.linalg.slogdet(Q)[1]
                    quad_terms = (-0.5 * np.trace(Q_inv @ yyT) -
                                  0.5 * np.trace(A.T @ Q_inv @ A @ xxT) -
                                  np.trace(A.T @ Q_inv @ yxT))
                    linear_terms = -np.dot(np.dot(A.T @ Q_inv, b), x) + np.dot(np.dot(Q_inv, b), y)
                    
                    transition_lp = vol_term + quad_terms + linear_terms
                    return transition_lp
                
                transition_params_sequence = (
                    compute_transition_params_sequence(drift_matrix, drift_bias, diffusion_scale, covariates[1:]))
                return vmap(_one_step_objective)(stats, transition_params_sequence) # should be (T,)
            
            objective_val = vmap(_single_element_objective)(batched_stats, batched_covariates) # should be (B, T)
            # import ipdb; ipdb.set_trace()
            return np.mean(objective_val)
           
        _objective(flattened_params)
        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            flattened_params,
            method="BFGS"
        )
        
        drift_matrix, drift_bias, diffusion_scale = unravel(optimize_results.x)
        self.drift_matrix = drift_matrix
        self.drift_bias = drift_bias
        self.diffusion_scale = diffusion_scale
        

