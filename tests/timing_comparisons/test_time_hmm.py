import tensorflow_probability.substrates.jax as tfp
import jax.random as jr
import jax.numpy as np
import pytest

from ssm.hmm import GaussianHMM, GaussianAutoregressiveHMM
from ssm.distributions.glm import GaussianLinearRegression

def make_random_hmm(emissions_dim=2, latent_dim=3, rng=jr.PRNGKey(0), emissions="gaussian"):
    num_states = latent_dim

    # # initial state
    initial_state_logits = np.zeros(num_states)
    initial_dist = tfp.distributions.Categorical(logits=initial_state_logits)

    # # dynamics
    transition_logits = np.zeros((num_states, num_states))
    transition_dist = tfp.distributions.Categorical(logits=transition_logits)

    # emissions
    if emissions == "gaussian":
        emission_means = 3 * jr.normal(rng, shape=(latent_dim, emissions_dim))
        emission_scale_trils = np.tile(np.eye(emissions_dim), (num_states, 1, 1))
        emission_dist = tfp.distributions.MultivariateNormalTriL(loc=emission_means, scale_tril=emission_scale_trils)
        return GaussianHMM(num_states, initial_dist, transition_dist, emission_dist)

    elif emissions == "autoregressive":
        emission_dist = GaussianLinearRegression(
            weights=np.tile(0.99 * np.eye(emissions_dim), (latent_dim, 1, 1)),
            bias=0.01 * jr.normal(rng, (latent_dim, emissions_dim)),
            scale_tril=np.tile(np.eye(emissions_dim), (latent_dim, 1, 1))
        )
        return GaussianAutoregressiveHMM(num_states, initial_dist, transition_dist, emission_dist)

def hmm_fit_em_setup(num_trials=5, num_timesteps=200, latent_dim=2, emissions_dim=10, num_iters=100, emissions="gaussian"):
    rng = jr.PRNGKey(0)
    true_rng, sample_rng, test_rng = jr.split(rng, 3)
    true_hmm = make_random_hmm(emissions_dim, latent_dim, true_rng, emissions=emissions)
    states, data = true_hmm.sample(sample_rng, num_timesteps, num_samples=num_trials)
    test_hmm = make_random_hmm(emissions_dim, latent_dim, test_rng, emissions=emissions)
    print("")  # for verbose pytest, this prevents tqdm from clobering pytest's layout
    return test_hmm, data, num_iters

def hmm_fit_em(hmm, data, num_iters):
    lp, fit_model, posteriors = hmm.fit(data, method="em", num_iters=num_iters, tol=-1)
    last_lp = lp[-1].block_until_ready  # explicitly block until ready (else jax has async dispatch)
    return lp


#### TEST GAUSSIAN HMM
class TestGaussianHMM:
    @pytest.mark.parametrize("num_trials", range(1, 202, 100))
    def test_hmm_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (hmm_fit_em_setup(num_trials=num_trials), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))
        
    @pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
    def test_hmm_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("latent_dim", range(2, 13, 2))
    def test_hmm_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("emissions_dim", range(2, 13, 2))
    def test_hmm_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))


#### TEST GAUSSIAN ARHMM
class TestGaussianARHMM:
    @pytest.mark.parametrize("num_trials", range(1, 202, 100))
    def test_arhmm_em_fit_num_trials(self, benchmark, num_trials):
        setup = lambda: (hmm_fit_em_setup(num_trials=num_trials, emissions="autoregressive"), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))
        
    @pytest.mark.parametrize("num_timesteps", range(10, 20011, 10000))
    def test_arhmm_em_fit_num_timesteps(self, benchmark, num_timesteps):
        setup = lambda: (hmm_fit_em_setup(num_timesteps=num_timesteps, emissions="autoregressive"), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("latent_dim", range(2, 13, 2))
    def test_arhmm_em_fit_latent_dim(self, benchmark, latent_dim):
        setup = lambda: (hmm_fit_em_setup(latent_dim=latent_dim, emissions="autoregressive"), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))

    @pytest.mark.parametrize("emissions_dim", range(2, 13, 2))
    def test_arhmm_em_fit_emissions_dim(self, benchmark, emissions_dim):
        setup = lambda: (hmm_fit_em_setup(emissions_dim=emissions_dim, emissions="autoregressive"), {})
        lp = benchmark.pedantic(hmm_fit_em, setup=setup, rounds=1)
        assert not np.any(np.isnan(lp))
