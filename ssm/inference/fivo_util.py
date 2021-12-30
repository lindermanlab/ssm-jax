"""
Utils for exploring FIVO..
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
from copy import deepcopy as dc
from timeit import default_timer as dt
import numpy as onp
import pickle as p
from types import SimpleNamespace
from typing import Iterable
import argparse
import git
import platform
from pprint import pprint

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.inference.smc import _plot_single_sweep
from ssm.inference.smc import smc
import ssm.utils as utils
import ssm.inference.fivo as fivo
from tensorflow_probability.substrates.jax import distributions as tfd

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
                         num_particles=10, dset_to_plot=0, init_model=None):
    """

    Args:
        key:
        true_model:
        dataset:
        true_states:
        opt:
        do_fivo_sweep_jitted:
        _smc_jit:
        num_particles:
        dset_to_plot:
        init_model:

    Returns:

    """

    # Do some sweeps.
    key, subkey = jr.split(key)
    smc_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
    key, subkey = jr.split(key)
    initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                num_particles=num_particles,
                                                                datasets=dataset)

    # CODE for plotting lineages.
    idx = 7
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
    for _p in smc_posterior[idx].weighted_smoothing_particles:
        ax[0].plot(_p, linewidth=0.1, c='b')
    ax[0].grid(True)
    for _p in sweep_posteriors[idx].weighted_smoothing_particles:
        ax[1].plot(_p, linewidth=0.1, c='b')
    ax[1].grid(True)
    plt.pause(0.01)

    # Compare the variances of the LML estimates.
    # Test BPF in the initial model..
    val_bpf_lml, val_fivo_lml = [], []
    for _ in range(20):
        key, subkey = jr.split(key)
        true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
        true_bpf_lml = - utils.lexp(true_bpf_posterior.log_normalizer)
        val_bpf_lml.append(true_bpf_lml)

    for _ in range(20):
        key, subkey = jr.split(key)
        initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                    num_particles=num_particles,
                                                                    datasets=dataset)
        initial_lml = -utils.lexp(sweep_posteriors.log_normalizer)
        val_fivo_lml.append(initial_lml)

    print('Variance: BPF:      ', np.var(np.asarray(val_bpf_lml)))
    print('Variance: FIVO-AUX: ', np.var(np.asarray(val_fivo_lml)))


def compute_marginal_kls(get_marginals, true_model, dataset, smoothing_particles):
    """

    E_q [ log Q / P ]

    Args:
        get_marginals:
        true_model:
        dataset:
        smoothing_particles:

    Returns:

    """

    eps = 1e-3

    # Get the analytic smoothing marginals.
    marginals = get_marginals(true_model, dataset)

    if marginals is None:
        # TODO - make this more reliable somehow.
        # If there was no analytic marginal available.
        return np.asarray([np.inf])

    # To compute the marginals we are just going to fit a Gaussian.
    kl_p_q = []
    for _t in range(smoothing_particles.shape[-2]):
        samples = smoothing_particles.squeeze()[:, :, _t]
        q_mu = np.mean(samples, axis=1)
        q_sd = np.std(samples, axis=1) + eps

        p_mu = marginals.mean()[:, _t]
        p_sd = marginals.stddev()[:, _t] + eps

        _kl_p_q = np.log(q_sd / p_sd) + \
                  (((p_sd ** 2) + ((p_mu - q_mu) ** 2)) / (2.0 * (q_sd ** 2))) + \
                  - 0.5

        kl_p_q.append(_kl_p_q)

    return np.asarray(kl_p_q)


def initial_validation(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
                       num_particles=1000, dset_to_plot=0, init_model=None, GLOBAL_PLOT=True, do_print=None):
    """

    Args:
        key:
        true_model:
        dataset:
        true_states:
        opt:
        do_fivo_sweep_jitted:
        _smc_jit:
        num_particles:
        dset_to_plot:
        init_model:

    Returns:

    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0
    init_bpf_posterior = None
    em_posterior = None
    true_bpf_posterior = None
    true_lml = 0.0
    initial_fivo_bound = 0.0
    init_smc_posterior = None
    initial_lml = 0.0

    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.e_step)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
    true_lml = - utils.lexp(true_bpf_posterior.log_normalizer)

    if init_model is not None:
        # Test BPF in the initial model..
        key, subkey = jr.split(key)
        init_bpf_posterior = _smc_jit(subkey, init_model, dataset, num_particles=num_particles)
        initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)
        print('Initial BPF LML: ', initial_bpf_lml)

    # Test SMC in the initial model.
    key, subkey = jr.split(key)
    initial_fivo_bound, init_smc_posterior = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                  _num_particles=num_particles,
                                                                  _datasets=dataset)
    initial_lml = -utils.lexp(init_smc_posterior.log_normalizer)

    # # Dump any odd and ends of test code in here.
    # temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
    #                      num_particles=10, dset_to_plot=dset_to_plot, init_model=init_model)

    # Do some plotting.
    if em_posterior is not None:
        sweep_em_mean = em_posterior.mean()[dset_to_plot]
        sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
        sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
        _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
                           tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])

    if true_bpf_posterior is not None:
        _plot_single_sweep(true_bpf_posterior[dset_to_plot].filtering_particles,
                           true_states[dset_to_plot],
                           tag='True BPF Filtering (' + str(num_particles) + ' particles).',
                           obs=dataset[dset_to_plot])
        _plot_single_sweep(true_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                           true_states[dset_to_plot],
                           tag='True BPF Smoothing (' + str(num_particles) + ' particles).',
                           obs=dataset[dset_to_plot])

    if init_bpf_posterior is not None:
        _plot_single_sweep(init_bpf_posterior[dset_to_plot].filtering_particles,
                           true_states[dset_to_plot],
                           tag='Initial BPF Filtering (' + str(num_particles) + ' particles).',
                           obs=dataset[dset_to_plot])
        _plot_single_sweep(init_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                           true_states[dset_to_plot],
                           tag='Initial BPF Smoothing (' + str(num_particles) + ' particles).',
                           obs=dataset[dset_to_plot])

    if init_smc_posterior is not None:
        filt_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].filtering_particles,
                                      true_states[dset_to_plot],
                                      tag='Initial SMC Filtering (' + str(num_particles) + ' particles).',
                                      obs=dataset[dset_to_plot])
        sweep_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
                                       true_states[dset_to_plot],
                                       tag='Initial SMC Smoothing (' + str(num_particles) + ' particles).',
                                       obs=dataset[dset_to_plot])

    # Do some print.
    if do_print is not None: do_print(0, true_model, opt, true_lml, initial_lml, initial_fivo_bound, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def compare_kls(get_marginals, env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted, plot=False, GLOBAL_PLOT=True):
    """

    Args:
        get_marginals:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:
        plot:

    Returns:

    """

    num_particles = env.config.sweep_test_particles

    # Compare the KLs of the smoothing distributions.
    key, subkey = jr.split(key)
    true_bpf_posterior = smc_jitted(subkey, true_model, dataset, num_particles=num_particles)
    key, subkey = jr.split(key)
    _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                 fivo.get_params_from_opt(opt),
                                                 _num_particles=num_particles,
                                                 _datasets=dataset)

    true_bpf_kls = compute_marginal_kls(get_marginals, true_model, dataset, true_bpf_posterior.weighted_smoothing_particles)
    pred_smc_kls = compute_marginal_kls(get_marginals, true_model, dataset, pred_smc_posterior.weighted_smoothing_particles)
    # init_bpf_kls = compute_marginal_kls(get_marginals, true_model, dataset, init_bpf_posterior.weighted_smoothing_particles)

    if plot and GLOBAL_PLOT:
        plt.figure()
        plt.plot(np.median(np.asarray(true_bpf_kls), axis=1), label='True (BPF)')
        plt.plot(np.median(np.asarray(pred_smc_kls), axis=1), label='Pred (FIVO-AUX)')
        # plt.plot(np.median(np.asarray(init_bpf_kls), axis=1), label='bpf')
        plt.legend()
        plt.grid(True)
        plt.title('E_sweeps [ KL [ p_true[t] || q_pred[t] ] ] (' + str(num_particles) + ').')
        plt.xlabel('Time, t')
        plt.ylabel('KL_t')
        plt.pause(0.001)
        plt.savefig('./figs/kl_diff.pdf')

    return true_bpf_kls, pred_smc_kls


def compare_sweeps(env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted,
                   tag='', nrep=10, true_states=None, num_particles=None):
    """

    Args:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:

    Returns:

    """
    if num_particles is None:
        num_particles = env.config.sweep_test_particles

    # # Do some final validation.
    # # Rebuild the initial distribution.
    # _prop = rebuild_prop_fn(fivo.get_params_from_opt(opt)[1])
    # if _prop is not None:
    #     initial_distribution = lambda _dset, _model:  _prop(_dset, _model, np.zeros(dataset.shape[-1], ), 0, _model.initial_distribution(), None)
    # else:
    #     initial_distribution = None

    # BPF in true model.
    key, subkey = jr.split(key)
    final_val_posterior_bpf_true = smc_jitted(subkey,
                                              true_model,
                                              dataset,
                                              num_particles=num_particles)

    # SMC with tilt.
    key, subkey = jr.split(key)
    _, final_val_posterior_fivo_aux = do_fivo_sweep_jitted(subkey,
                                                           fivo.get_params_from_opt(opt),
                                                           _num_particles=num_particles,
                                                           _datasets=dataset,)
    # final_val_posterior_fivo_aux = smc(subkey,
    #                                           rebuild_model_fn(fivo.get_params_from_opt(opt)[0]),
    #                                           dataset,
    #                                           initialization_distribution=initial_distribution,
    #                                           proposal=rebuild_prop_fn(fivo.get_params_from_opt(opt)[1]),
    #                                           tilt=rebuild_tilt_fn(fivo.get_params_from_opt(opt)[2]),
    #                                           num_particles=num_particles)

    # CODE for plotting lineages.
    for _dset_idx in range(nrep):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
        plt.suptitle('Tag: ' + str(tag) + ', ' + str(num_particles) + ' particles.')

        for _p in final_val_posterior_bpf_true[_dset_idx].weighted_smoothing_particles:
            ax[0].plot(_p, linewidth=0.1, c='b')
        ax[0].grid(True)
        ax[0].set_title('BPF in true model')

        for _p in final_val_posterior_fivo_aux[_dset_idx].weighted_smoothing_particles:
            ax[1].plot(_p, linewidth=0.1, c='b')
        ax[1].grid(True)

        if (fivo.get_params_from_opt(opt)[1] is not None) and (fivo.get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pqr.')
            _tag = 'pqr'
        elif (fivo.get_params_from_opt(opt)[1] is not None) and (fivo.get_params_from_opt(opt)[2] is None):
            ax[1].set_title('SMC-AUX with learned pq.')
            _tag = 'pq'
        elif (fivo.get_params_from_opt(opt)[1] is None) and (fivo.get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pr.')
            _tag = 'pr'
        else:
            ax[1].set_title('SMC-AUX with learned p...?')
            _tag = 'p'

        if true_states is not None:
            ax[0].plot(true_states[_dset_idx], linewidth=0.25, c='k', linestyle=':')
            ax[1].plot(true_states[_dset_idx], linewidth=0.25, c='k', linestyle=':')

        plt.pause(0.01)
        plt.savefig('./figs/tmp_sweep_{}_{}.pdf'.format(_tag, _dset_idx))
        plt.close(fig)


def final_validation(get_marginals,
                     env,
                     opt, 
                     dataset, 
                     true_model, 
                     rebuild_model_fn, 
                     rebuild_prop_fn, 
                     rebuild_tilt_fn, 
                     key, 
                     do_fivo_sweep_jitted,
                     smc_jitted,
                     GLOBAL_PLOT=True,
                     tag=''):
    """

    Args:
        get_marginals:
        env:
        opt:
        dataset:
        true_model:
        rebuild_model_fn:
        rebuild_prop_fn:
        rebuild_tilt_fn:
        key:
        do_fivo_sweep_jitted:

    Returns:

    """

    # Compare the sweeps.
    compare_sweeps(get_marginals, env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted, tag=tag)

    # Compare the KLs.
    true_bpf_kls, pred_smc_kls = compare_kls(get_marginals, env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn,
                                             rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted, plot=True, GLOBAL_PLOT=GLOBAL_PLOT)

