import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

import numpy as onp

from ssm.twarhmm import GaussianTWARHMM
from ssm.utils import random_rotation

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x

#### TESTS

def test_twarhmm_jit():
    time_constants = np.logspace(-1, 1, num=25, base=4)
    twarhmm = GaussianTWARHMM(5, time_constants, 3, seed=jr.PRNGKey(0))
    identity(twarhmm)

def test_twarhmm_sample():
    rng1, rng2 = jr.split(SEED, 2)
    time_constants = np.logspace(-1, 1, num=25, base=4)
    twarhmm = GaussianTWARHMM(5, time_constants, 3, seed=rng1)
    states, data = twarhmm.sample(rng2, num_steps=10, num_samples=32)
    states, time_constants = states
    assert states.shape == (32, 10)
    assert time_constants.shape == (32, 10)
    assert data.shape == (32, 10, 3)

def test_twarhmm_sample_is_consistent():
    # sampled from FHMM previously
    true_states = np.array([[2, 1, 1, 1, 1, 4, 4, 4, 4, 2],
                            [4, 4, 4, 1, 2, 2, 2, 2, 2, 0]], dtype=np.int32)
    true_time_constants = np.array([[20, 20, 20, 20, 20,  6,  6,  6,  6,  6],
                                    [11, 11, 11, 11, 11, 11,  5,  5,  5,  5]], dtype=np.int32)
    true_data = np.array([[[-0.5789567 ,  0.01260954],
                           [-2.0834973 , -0.18862861],
                           [-2.8359954 , -0.13907951],
                           [-2.1441557 ,  0.7015305 ],
                           [-1.0234834 ,  1.186448  ],
                           [ 0.27287763,  2.801372  ],
                           [-0.70958316,  2.888864  ],
                           [ 0.99850607,  2.8689687 ],
                           [ 1.0432734 ,  2.384862  ],
                           [ 0.4378001 ,  2.3896494 ]],
                          [[-0.32371512, -0.31010967],
                           [ 1.333456  ,  1.3727825 ],
                           [-1.8606696 ,  1.1155223 ],
                           [-2.3177795 ,  1.7310637 ],
                           [-2.3041327 ,  2.171277  ],
                           [-1.3990531 ,  2.5384762 ],
                           [ 1.2161027 ,  3.414078  ],
                           [ 0.4189844 ,  6.308393  ],
                           [ 0.5240829 ,  4.100033  ],
                           [-1.0810217 ,  4.888917  ]]],
                         dtype=np.float32)

    rng1, rng2 = jr.split(SEED, 2)
    time_constants = np.logspace(-1, 1, num=25, base=4)
    twarhmm = GaussianTWARHMM(5, time_constants, 2, seed=rng1)
    states, data = twarhmm.sample(rng2, num_steps=10, num_samples=2)
    states, sampled_time_constants = states
    assert np.array_equal(states, true_states)
    assert np.array_equal(sampled_time_constants, true_time_constants)
    assert np.allclose(true_data, data, atol=1e-5)

def test_twarhmm_em_fit():
    rng1, rng2 = jr.split(SEED, 2)
    
    # right now, fit on random TWARHMM results in NaNs
    # we need to explcitly construct weights, biases to avoid this
    # TODO: investigate this behavior
    
    num_states = 3
    data_dim = 2
    num_lags = 1
    
    # construct transition matrix
    transition_probs = (np.arange(num_states)**10).astype(float)
    transition_probs /= transition_probs.sum()
    transition_matrix = np.zeros((num_states, num_states))
    for k, p in enumerate(transition_probs[::-1]):
        transition_matrix += np.roll(p * np.eye(num_states), k, axis=1)
    
    # construct weights, biases, and cov
    keys = jr.split(rng1, num_states)
    angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False)
    theta = np.pi / 25 # rotational frequency
    weights = np.array([0.8 * random_rotation(key, data_dim, theta=theta) for key in keys])
    biases = np.column_stack([np.cos(angles), np.sin(angles), np.zeros((num_states, data_dim - 2))])
    covariances = np.tile(0.001 * np.eye(data_dim), (num_states, 1, 1))
    
    # time constant possibilities
    time_constants = np.logspace(-1, 1, num=25, base=4)
    
    # Make a Time-Warped Autoregressive (TWAR)HMM
    twarhmm = GaussianTWARHMM(num_states, 
                              time_constants,
                              discrete_state_transition_matrix=transition_matrix,
                              emission_weights=weights - np.eye(data_dim),
                              emission_biases=biases,
                              emission_covariances=covariances)
    
    states, data = twarhmm.sample(rng2, num_steps=1000, num_samples=1)
    test_twarhmm = GaussianTWARHMM(num_states, time_constants, data_dim, seed=rng2)
    
    # fit with no early stopping
    lp, fitted_twarhmm, posteriors = test_twarhmm.fit(data, num_iters=100, tol=-1)

    # some simple tests
    assert not np.any(np.isnan(lp))
    assert posteriors.expected_states.shape == (1, 1000, 3, 25)
