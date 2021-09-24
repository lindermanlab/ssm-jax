from collections import namedtuple

import jax.numpy as np
from jax import jit, value_and_grad, hessian
from jax.scipy.linalg import solve_triangular
from jax import lax

import jax.experimental.optimizers as optimizers

from ssm.models.lds import GaussianLDS
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

# TODO: @schlagercollin consider Class for laplace EM fit
class StructuredMeanFieldVariationalPosterior:
    def __init__(self):
        return NotImplementedError
    
def _laplace_log_joint(lds: GaussianLDS, x, data):
    """[summary]

    Args:
        lds ([type]): [description]
        x ([type]): [description]
        data ([type]): [description]
    """
    return lds.log_probability(x, data)
    
def _fit_laplace_find_mode(lds, x0, data, learning_rate=1e-3, num_iters=1500):
    """Find the mode of the log joint for the Laplace approximation.
    
    Args:
        lds ([type]): [description]
        x0 ([type]): [description]
        data ([type]): [description]
        learning_rate ([type], optional): [description]. Defaults to 1e-3.

    Returns:
        [type]: [description]
    """
    
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(x0)
    scale = x0.size
    
    def _objective(x):
        return -1 * _laplace_log_joint(lds=lds, x=x, data=data)
    
    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(_objective)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state
    
    # guaranteed to convergence?
    for i in range(num_iters):
        value, opt_state = step(i, opt_state)
        
    return get_params(opt_state)


def _laplace_neg_hessian_log_dynamics(lds, data, x):
    """[summary]

    Args:
        lds ([type]): [description]
        data ([type]): [description]
        x ([type]): [description]
    """
    return NotImplementedError
    # # initial distribution contributes a Gaussian term to first diagonal block
    # J_ini = np.sum(Ez[0, :, None, None] * np.linalg.inv(self.Sigmas_init), axis=0)

    # # first part is transition dynamics - goes to all terms except final one
    # # E_q(z) x_{t} A_{z_t+1}.T Sigma_{z_t+1}^{-1} A_{z_t+1} x_{t}
    # inv_Sigmas = np.linalg.inv(self.Sigmas)
    # dynamics_terms = np.array([A.T@inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)]) # A^T Qinv A terms
    # J_dyn_11 = np.sum(Ez[1:,:,None,None] * dynamics_terms[None,:], axis=1)

    # # second part of diagonal blocks are inverse covariance matrices - goes to all but first time bin
    # # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} x_{t+1}
    # J_dyn_22 = np.sum(Ez[1:,:,None,None] * inv_Sigmas[None,:], axis=1)

    # # lower diagonal blocks are (T-1,D,D):
    # # E_q(z) x_{t+1} Sigma_{z_t+1}^{-1} A_{z_t+1} x_t
    # off_diag_terms = np.array([inv_Sigma@A for A, inv_Sigma in zip(self.As, inv_Sigmas)])
    # J_dyn_21 = -1 * np.sum(Ez[1:,:,None,None] * off_diag_terms[None,:], axis=1)

    # return J_ini, J_dyn_11, J_dyn_21, J_dyn_22
    
def _laplace_neg_hessian_log_observations(lds: GaussianLDS, data, x):
    """[summary]

    Args:
        lds ([type]): [description]
        data ([type]): [description]
        x ([type]): [description]
    """
    obj = lambda x, data: lds.emissions_distribution(x).log_prob(data)
    hess = hessian(obj)
    return -1 * hess(x, data)
    

def _laplace_neg_hessian_params(lds, data, x):
    """Evaluate Hessian of log joint at the mode (x).
    """
    # TODO: can we just get this using a natural_parameters() method?
    
    J_ini, J_dyn_11, J_dyn_22 = _laplace_neg_hessian_log_dynamics(lds, data, x)
    J_obs = _laplace_neg_hessian_log_observations(lds, data, x)
    J = (J_ini, J_dyn_11, J_dyn_22, J_obs)
    h = _compute_h_from_J_params(*J)
    
    return J, h


def _fit_laplace_continuous_state_update(lds, variational_posterior, data):
    """Update q(x) using a Laplace approximation.
    
    Find argmax of the log posterior
    - compute gradient g(x) and block tridiagonal Hessian J(x)
    - update: x' = x + J(x)^{-1} g(x)
    - check for convergence
    
    Evaluate the J(x*) at the optimal x*.

    Args:
        variational_posterior ([type]): [description]
        data ([type]): [description]
    """
    
    # Find mode of expected log joint
    x0 = variational_posterior.mean_continuous_states
    x = _fit_laplace_find_mode(lds, x0, data)
    
    # Evaluate Hessian at the mode (to get J, h for Laplace approx)
    J, h = _laplace_neg_hessian_params(lds, data, x)
    
    # Update the variational posterior parameters
    variational_posterior.update_parameters(J, h)    


def _fit_laplace_em(lds,
                    variational_posterior,
                    data,
                    num_iters=100,
                    num_samples=1,
                    verbose=2):
    """[summary]

    Args:
        variational_posterior ([type]): [description]
        data ([type]): [description]
        num_iters (int, optional): [description]. Defaults to 100.
        num_samples (int, optional): [description]. Defaults to 1.
    """
    
    elbos = [_laplace_em_elbo(variational_posterior, data)]
    pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.1f}", [elbos[-1]])
    
    for iter in pbar:
        
        # 1. Update q(x) using Laplace approximation (E Step)
        _fit_laplace_continuous_state_update(lds, variational_posterior, data)
        
        # 2. Update parameters using samples from q(x) (M Step)
        _fit_laplace_params_update(lds, variational_posterior, data)
        
        # Compute the evidence lower bound
        elbos.append(_laplace_em_elbo(lds, variational_posterior, data))
        
        if verbose == 2:
            pbar.set_description("ELBO {:.1f}".format(elbos[-1]))
        
    return np.array(elbos), lds