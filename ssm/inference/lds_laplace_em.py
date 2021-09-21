from collections import namedtuple

import jax.numpy as np
from jax import jit, value_and_grad
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
    
def _laplace_neg_expected_log_joint(lds: GaussianLDS, x, data):
    """[summary]

    Args:
        lds ([type]): [description]
        x ([type]): [description]
        data ([type]): [description]
    """
    return NotImplementedError
    
def _fit_laplace_find_mode(x0, data, learning_rate=1e-3):
    """Find the mode of the expected log joint for Laplace approx.

    Args:
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
        return _laplace_neg_expected_log_joint(x=x, data=data)
    
    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(_objective)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state
    
    for i in range(1500):
        value, opt_state = step(i, opt_state)
        
    return get_params(opt_state)

def _fit_laplace_continuous_state_update(variational_posterior, data):
    """Update q(x) using a Laplace approximation
    
    Find argmax of the expected log joint
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
    x = _fit_laplace_find_mode(x0, data)
    
    # Evaluate Hessian at the mode (to get J, h for Laplace approx)
    J, h = _laplace_neg_hessian_params(data, x)
    
    # Update the variational posterior parameters
    variational_posterior.update_parameters(J, h)
    


def _fit_laplace_em(variational_posterior,
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
        
        # 1. Update q(x) using Laplace approximation
        _fit_laplace_continuous_state_update(variational_posterior, data)
        
        # 2. Update parameters using samples from q(x)
        _fit_laplace_params_update(variational_posterior, data)
        
        # Compute the evidence lower bound
        elbos.append(_laplace_em_elbo(variational_posterior, data))
        
        if verbose == 2:
            pbar.set_description("ELBO {:.1f}".format(elbos[-1]))
        
    return np.array(elbos)