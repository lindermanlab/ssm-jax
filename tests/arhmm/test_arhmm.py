import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

from ssm.arhmm import GaussianARHMM

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x 

#### TESTS

def test_gaussian_arhmm_jit():
    arhmm = GaussianARHMM(3, 5, 1, seed=SEED)
    identity(arhmm)
    
def test_gaussian_arhmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    arhmm = GaussianARHMM(3, 5, 1, seed=rng1)
    states, data = arhmm.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10)
    assert data.shape == (32, 10, 5)
    
def test_gaussian_arhmm_sample_is_consistent():
    # sampled from ARHMM previously 
    true_states = np.array([[0, 2, 1],
                            [0, 0, 0]], dtype=np.int32)
    true_data = np.array([[[-1.0043284 ,  0.4558727 ],
                           [ 4.4478188 , -1.7673836 ],
                           [-9.901798  ,  0.79967743]],
                          [[-0.4406956 ,  0.06618765],
                           [-1.059847  ,  0.6474356 ],
                           [ 2.0843577 ,  0.72555685]]], dtype=np.float32)
    
    rng1, rng2 = jr.split(SEED, 2)
    arhmm = GaussianARHMM(3, 2, 1, seed=rng1)
    states, data = arhmm.sample(rng2, num_steps=3, num_samples=2)
    assert np.all(true_states == states)
    assert np.allclose(true_data, data, atol=1e-5)
    
def test_gaussian_arhmm_em_fit():
    rng1, rng2, rng3 = jr.split(SEED, 3)

    # TODO: can we just use a random ARHMM? [else NaNs in fit]
    def createNiceGaussianARHMM(num_states, emissions_dim, num_lags, seed):
        emission_weights = np.tile(0.99 * np.eye(emissions_dim), (num_states, 1, 1))
        emission_biases = 0.01 * jr.normal(seed, (num_states, emissions_dim))
        emission_covariances = np.tile(np.eye(emissions_dim), (num_states, 1, 1))
        return GaussianARHMM(
            num_states,
            num_lags=num_lags,
            emission_weights=emission_weights,
            emission_biases=emission_biases,
            emission_covariances=emission_covariances,
        )

    true_arhmm = createNiceGaussianARHMM(3, 5, 1, seed=rng1)
    states, data = true_arhmm.sample(rng2, num_steps=100, num_samples=3)
    test_arhmm = GaussianARHMM(3, 5, 1, seed=rng3)
    
    # fit with no early stopping 
    lp, fitted_hmm, posteriors = test_arhmm.fit(data, num_iters=100, tol=-1)
    
    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (3, 100, 3)
