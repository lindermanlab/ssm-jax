import pytest
import numpy as np

from ssm import HMM

import sys

sys.path.append("..")
import config


def sample_hmm(true_hmm, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_hmm.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data


def hmm_fit_em_setup(
    num_trials=config.NUM_TRIALS,
    num_timesteps=config.NUM_TIMESTEPS,
    latent_dim=config.LATENT_DIM,
    emissions_dim=config.EMISSIONS_DIM,
    num_iters=config.NUM_ITERS,
    emissions="gaussian",
):
    params = locals()
    true_hmm = HMM(emissions_dim, latent_dim, dynamics="gaussian", emissions=emissions)
    states, data = sample_hmm(true_hmm, num_trials=num_trials, time_bins=num_timesteps)
    test_hmm = HMM(emissions_dim, latent_dim, dynamics="gaussian", emissions=emissions)
    return test_hmm, data, num_iters, params


def hmm_fit_em(hmm, data, num_iters, params):
    lps = hmm.fit(
        datas=data, method="em", num_init_iters=0, num_iters=num_iters, tolerance=-1
    )
    return lps, params


def run_time_test(benchmark, time_fn, setup_fn):
    lp, params = benchmark.pedantic(time_fn, setup=setup_fn, rounds=config.NUM_ROUNDS)
    benchmark.extra_info["params"] = params
    assert not np.any(np.isnan(lp))


#### GAUSSIAN EM TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
def test_hmm_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (hmm_fit_em_setup(num_trials=num_trials), {})
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
def test_hmm_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps), {})
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
def test_hmm_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim), {})
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
def test_hmm_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim), {})
    run_time_test(benchmark, hmm_fit_em, setup)


#### ARHMM TESTS
#### TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
def test_arhmm_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (
        hmm_fit_em_setup(num_trials=num_trials, emissions="autoregressive"),
        {},
    )
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP)
def test_arhmm_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (
        hmm_fit_em_setup(num_timesteps=num_timesteps, emissions="autoregressive"),
        {},
    )
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
def test_arhmm_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (
        hmm_fit_em_setup(latent_dim=latent_dim, emissions="autoregressive"),
        {},
    )
    run_time_test(benchmark, hmm_fit_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
def test_arhmm_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (
        hmm_fit_em_setup(emissions_dim=emissions_dim, emissions="autoregressive"),
        {},
    )
    run_time_test(benchmark, hmm_fit_em, setup)
