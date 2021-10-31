import pytest
import numpy as np

from ssm import HMM

from ..config import NUM_ROUNDS

def sample_hmm(true_hmm, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_hmm.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data

def hmm_fit_em_setup(num_trials=5, num_timesteps=100, latent_dim=3, emissions_dim=10,
                     num_iters=100, emissions="gaussian"):
    true_hmm = HMM(emissions_dim, latent_dim, dynamics="gaussian", emissions=emissions)
    states, data = sample_hmm(true_hmm, num_trials=num_trials, time_bins=num_timesteps)
    test_hmm = HMM(emissions_dim, latent_dim, dynamics="gaussian", emissions=emissions)
    return test_hmm, data, num_iters

def hmm_fit_em(hmm, data, num_iters):
    lps = hmm.fit(datas=data, method="em", num_init_iters=0, num_iters=num_iters, tolerance=-1)
    return lps


#### GAUSSIAN EM TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", range(1, 202, 100))
def test_hmm_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (hmm_fit_em_setup(num_trials=num_trials), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))
    
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
def test_hmm_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", range(2, 13, 2))
def test_hmm_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", range(2, 13, 2))
def test_hmm_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))

#### ARHMM TESTS
#### TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", range(1, 202, 100))
def test_arhmm_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (hmm_fit_em_setup(num_trials=num_trials, emissions="autoregressive"), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))
    
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
def test_arhmm_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps, emissions="autoregressive"), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", range(2, 13, 2))
def test_arhmm_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim, emissions="autoregressive"), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))

@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", range(2, 13, 2))
def test_arhmm_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim, emissions="autoregressive"), {})
    lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=NUM_ROUNDS)
    assert not np.any(np.isnan(lp))



