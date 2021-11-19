import pytest

import ssm.ctlds.dynamics as dynamics
import jax.random as jr
import jax.numpy as np

import numpy as onp

from jax.scipy.linalg import expm

def test_dynamics_distribution():
    latent_dim = 2
    
    drift_matrix = np.eye(latent_dim)
    scale = np.eye(latent_dim)
    drift_bias = np.zeros((latent_dim,))

    myDynamics = dynamics.StationaryDynamics(drift_matrix, drift_bias, scale)
    x = np.ones((latent_dim,))
    covariates = 5

    dist = myDynamics.distribution(x, covariates)
    
    true_mean = expm(drift_matrix * 5.) @ x
    onp.testing.assert_allclose(dist.mean(), true_mean)

    
    

