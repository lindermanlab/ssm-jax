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
