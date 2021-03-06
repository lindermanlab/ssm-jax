from time import time

import pytest

import matplotlib.pyplot as plt
import numpy as np
import ssm
from ssm import LDS
from tqdm.auto import trange

import sys

sys.path.append("..")
import config


def sample_lds(true_lds, num_trials, time_bins):
    all_states, all_data = [], []
    for i in range(num_trials):
        states, data = true_lds.sample(T=time_bins)
        all_states.append(states)
        all_data.append(data)
    return all_states, all_data


def lds_fit_laplace_em_setup(
    num_trials=config.NUM_TRIALS,
    num_timesteps=config.NUM_TIMESTEPS,
    latent_dim=config.LATENT_DIM,
    emissions_dim=config.EMISSIONS_DIM,
    num_iters=config.NUM_ITERS,
):
    N = emissions_dim
    D = latent_dim
    true_lds = LDS(N, D, dynamics="gaussian", emissions="poisson")
    states, data = sample_lds(true_lds, num_trials=num_trials, time_bins=num_timesteps)
    test_lds = LDS(N, D, dynamics="gaussian", emissions="poisson")
    return test_lds, data, num_iters


def lds_fit_laplace_em(test_lds, data, num_iters):
    q_elbos_lem, q_lem = test_lds.fit(
        datas=data,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_init_iters=0,
        num_iters=num_iters,
        verbose=False,
    )
    return q_elbos_lem


def run_time_test(benchmark, time_fn, setup_fn):
    lp, params = benchmark.pedantic(time_fn, setup=setup_fn, rounds=config.NUM_ROUNDS)
    benchmark.extra_info["params"] = params
    assert not np.any(np.isnan(lp))


#### TESTS
@pytest.mark.ssmv0
@pytest.mark.parametrize("num_trials", config.NUM_TRIALS_SWEEP)
def test_lds_laplace_em_fit_num_trials(benchmark, num_trials):
    setup = lambda: (lds_fit_laplace_em_setup(num_trials=num_trials), {})
    run_time_test(benchmark, lds_fit_laplace_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("num_timesteps", config.NUM_TIMESTEPS_SWEEP_LDS_SSM_V0)
def test_lds_laplace_em_fit_num_timesteps(benchmark, num_timesteps):
    setup = lambda: (lds_fit_laplace_em_setup(num_timesteps=num_timesteps), {})
    run_time_test(benchmark, lds_fit_laplace_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("latent_dim", config.LATENT_DIM_SWEEP)
def test_lds_laplace_em_fit_latent_dim(benchmark, latent_dim):
    setup = lambda: (lds_fit_laplace_em_setup(latent_dim=latent_dim), {})
    run_time_test(benchmark, lds_fit_laplace_em, setup)


@pytest.mark.ssmv0
@pytest.mark.parametrize("emissions_dim", config.EMISSIONS_DIM_SWEEP)
def test_lds_laplace_em_fit_emissions_dim(benchmark, emissions_dim):
    setup = lambda: (lds_fit_laplace_em_setup(emissions_dim=emissions_dim), {})
    run_time_test(benchmark, lds_fit_laplace_em, setup)
