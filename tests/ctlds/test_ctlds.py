import pytest

import ssm.ctlds.dynamics as dynamics
from ssm.ctlds import GaussianCTLDS
import jax.random as jr
import jax.numpy as np
from jax import vmap

import numpy as onp

import jax.scipy.linalg as jspla

SEED = jr.PRNGKey(0)

def test_dynamics_distribution():
    latent_dim = 2
    
    drift_matrix = np.eye(latent_dim)
    scale = np.eye(latent_dim)
    drift_bias = np.zeros((latent_dim,))

    myDynamics = dynamics.StationaryCTDynamics(drift_matrix, drift_bias, scale)
    x = np.ones((latent_dim,))
    covariates = 5

    dist = myDynamics.distribution(x, covariates)
    
    true_mean = jspla.expm(drift_matrix * 5.) @ x
    onp.testing.assert_allclose(dist.mean(), true_mean)

def test_compute_sequence_transition_params():
    seed = jr.PRNGKey(0)
    num_timesteps = 5
    latent_dim = 2
    
    drift_matrix = np.eye(latent_dim)
    scale = np.eye(latent_dim)
    drift_bias = np.zeros((latent_dim,))
    
    covariates = jr.uniform(seed, (num_timesteps,))
    
    compute_sequence_transition_params = vmap(dynamics.compute_transition_params, in_axes=(None, None, None, 0))
    sequence_transition_params = compute_sequence_transition_params(drift_matrix, drift_bias, scale, covariates)
    assert len(sequence_transition_params) == 3
    assert sequence_transition_params[0].shape == (num_timesteps, latent_dim, latent_dim)
    assert sequence_transition_params[1].shape == (num_timesteps, latent_dim)
    assert sequence_transition_params[2].shape == (num_timesteps, latent_dim, latent_dim)
    
def test_gaussian_ctlds_em_fit():
    num_samples = 3
    num_steps = 100
    latent_dim = 2
    data_dim = 4
    
    rng1, rng2, rng3, rng4 = jr.split(SEED, 4)
    true_lds = GaussianCTLDS(latent_dim, data_dim, seed=rng1)
    covariates = jr.uniform(rng2, shape=(num_samples, num_steps)) + 0.1
    states, data = true_lds.sample(rng3, covariates=covariates, num_steps=num_steps, num_samples=num_samples)
    test_lds = GaussianCTLDS(latent_dim, data_dim, seed=rng4)
    
    # fit with no early stopping 
    lp, fitted_model, posteriors = test_lds.fit(data, covariates=covariates, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
