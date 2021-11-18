import pytest

import jax.random as jr
import jax.numpy as np
from jax import jit

from ssm.lds import GaussianLDS, PoissonLDS

SEED = jr.PRNGKey(0)

@jit
def identity(x):
    return x 

#### TESTS

def test_gaussian_lds_jit():
    lds = GaussianLDS(3, 5, seed=SEED)
    identity(lds)
    
def test_poisson_lds_jit():
    lds = PoissonLDS(3, 5, seed=SEED)
    identity(lds)
    
def test_gaussian_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = GaussianLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
    
def test_poisson_lds_sample():
    rng1, rng2 = jr.split(SEED, 2)
    lds = PoissonLDS(3, 5, seed=rng1)
    states, data = lds.sample(rng2, num_steps=10, num_samples=32)
    assert states.shape == (32, 10, 3)
    assert data.shape == (32, 10, 5)
