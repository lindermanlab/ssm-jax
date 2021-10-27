from time import time

import pytest

import matplotlib.pyplot as plt
import numpy as np
import ssm
from ssm import LDS
from tqdm.auto import trange

def sample_lds(true_lds, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_lds.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data

def lds_fit_em_setup(num_trials=5, num_timesteps=100, latent_dim=3, emissions_dim=10, num_iters=100):
    N = emissions_dim
    D = latent_dim    
    true_lds = LDS(N, D, dynamics="gaussian", emissions="gaussian")
    states, data = sample_lds(true_lds, num_trials=num_trials, time_bins=num_timesteps)
    test_lds = LDS(N, D, dynamics="gaussian", emissions="gaussian")
    return test_lds, data, num_iters

def lds_fit_em(test_lds, data, num_iters):
    q_elbos_lem, q_lem = test_lds.fit(
        datas=data,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_init_iters=0, num_iters=num_iters, verbose=False
    )
    return q_elbos_lem


#### TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", range(1, 202, 50))
def test_lds_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (lds_fit_em_setup(num_trials=num_trials), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))
    
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", range(10, 1011, 250))
def test_lds_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (lds_fit_em_setup(num_timesteps=num_timesteps), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", range(2, 8, 5))
def test_lds_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (lds_fit_em_setup(latent_dim=latent_dim), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", range(2, 8, 5))
def test_lds_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (lds_fit_em_setup(emissions_dim=emissions_dim), {})
    lp = benchmark.pedantic(lds_fit_em, setup=setup, rounds=1)
    assert not np.any(np.isnan(lp))