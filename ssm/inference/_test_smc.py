"""
Test code for SMC filtering/smoothing for SSMs.
"""
import jax
import warnings
import jax.numpy as jnp
from jax import jit, vmap
from ssm.utils import Verbosity, format_dataset, ssm_pbar

# Specific imports for here.
import jax.scipy as jscipy
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy as dc
import matplotlib.pyplot as plt

from ssm.inference.smc import *
from ssm.lds.models import GaussianLDS


disable_jit = True
from contextlib import contextmanager, ExitStack
@contextmanager
def nothing():
    yield
possibly_disabled = jax.disable_jit if disable_jit else nothing


# Define the standard plotting colours.
color_names = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple"
]


def plot_single_sweep(particles, true_states, tag=''):

    gen_label = lambda _k, _s: _s if _k == 0 else None

    single_sweep_median = jnp.median(particles, axis=1)
    single_sweep_lsd = jnp.quantile(particles, 0.17, axis=1)
    single_sweep_usd = jnp.quantile(particles, 0.83, axis=1)

    ts = jnp.arange(len(true_states))

    plt.figure()

    for _i, _c in zip(range(single_sweep_median.shape[1]), color_names):
        plt.plot(ts, single_sweep_median[:, _i], c=_c, label=gen_label(_i, 'Predicted'))
        plt.fill_between(ts, single_sweep_lsd[:, _i], single_sweep_usd[:, _i], color=_c, alpha=0.1)

        plt.plot(ts, true_states[:, _i], c=_c, linestyle='--', label=gen_label(_i, 'True'))

    plt.title(tag)
    plt.grid(True)
    plt.tight_layout()
    # plt.xlim(0, 5)
    plt.legend()
    plt.pause(0.1)


def test_smc(key):
    r"""
    Script for deploying and inspecting SMC code.
    :return:
    """

    # Set up true model and draw some data.
    latent_dim = 2
    emissions_dim = 3
    num_trials = 5
    num_timesteps = 10
    num_particles = 100

    key, subkey = jax.random.split(key)
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, emission_scale_tril=0.25**2 * jnp.eye(emissions_dim))

    key, subkey = jax.random.split(key)
    true_states, data = model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # _do_plot(true_states, data)

    # Test against EM (which for the LDS is exact.
    em_posterior = jax.vmap(model.infer_posterior)(data)
    em_log_marginal_likelihood = model.marginal_likelihood(data, posterior=em_posterior)

    # Do the SMC filtering sweep using BPF.
    with possibly_disabled():
        repeat_smc = lambda _k: smc(_k, model, data, proposal=None, num_particles=num_particles)
        n_reps = 1

        key, subkey = jax.random.split(key)
        smoothing_particles, log_marginal_likelihood, ancestry, filtering_particles = jax.vmap(repeat_smc)(jax.random.split(subkey, num=n_reps))

        # Print the estimated marginals.
        for _smc, _em in zip(log_marginal_likelihood.T, em_log_marginal_likelihood):
            for __smc in _smc:
                print('SMC/EM LML: \t {: >6.4f} \t {: >6.4f}'.format(__smc, _em))
            print()

    # # Plot the results.
    rep = 0
    dset = 0
    sweep_filtering = filtering_particles[rep, dset]
    sweep_smoothing = smoothing_particles[rep, dset]
    sweep_em_mean = em_posterior.mean()[dset]
    sweep_em_sds = jnp.sqrt(jnp.asarray([[jnp.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset]
    sweep_true = true_states[dset]

    # SMC.
    plot_single_sweep(sweep_filtering, sweep_true, tag='SMC filtering')
    plot_single_sweep(sweep_smoothing, sweep_true, tag='SMC smoothing')

    # Now plot the EM results.
    x = range(sweep_true.shape[0])
    plt.figure()
    for _idx, _c in zip(range(len(sweep_em_mean.T)), color_names):
        plt.plot(x, sweep_em_mean[:, _idx], c=_c)
        plt.fill_between(x, sweep_em_mean[:, _idx] - sweep_em_sds[:, _idx], sweep_em_mean[:, _idx] + sweep_em_sds[:, _idx], color=_c, alpha=0.1)
        plt.plot(x, sweep_true[:, _idx], c=_c, linestyle='--')
    plt.grid(True)
    plt.title('EM Smoothing')
    plt.pause(0.001)
    p = 0


if __name__ == '__main__':
    print('Test SMC code.')
    _key = jax.random.PRNGKey(1)
    test_smc(_key)

    plt.waitforbuttonpress()

    print('Done testing.')
