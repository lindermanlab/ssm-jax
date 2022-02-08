import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax import random as jr
from copy import deepcopy as dc
import flax
import pickle

# Import some ssm stuff.
import ssm.utils as utils
from ssm.inference.smc import _plot_single_sweep
from ssm.inference.fivo import get_params_from_opt
from ssm.utils import Verbosity
from copy import deepcopy as dc
from tensorflow_probability.substrates.jax import distributions as tfd
from flax import optim
from flax import linen as nn
import ssm.nn_util as nn_util

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# Define the standard plotting colours.
color_names = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan"
] * 10


def log_params(_param_hist, _cur_params):
    """
    Parse the parameters and store them for printing.

    Args:
        _param_hist:
        _cur_params:

    Returns:

    """
    failed_str = ''

    # MODEL.
    try:
        if _cur_params[0] is not None:
            _p = _cur_params[0]._asdict()
            _p_flat = {}
            for _k in _p.keys():
                _p_flat[_k] = dc(onp.array(_p[_k].flatten()))
            _param_hist[0].append(_p_flat)
        else:
            _param_hist[0].append(None)
    except:
        _param_hist[0].append(None)
        failed_str += 'Model, '

    # PROPOSAL.
    try:
        if _cur_params[1] is not None:
            _p = _cur_params[1]['params']._dict
            _p_flat = {}
            for _ko in _p.keys():
                for _ki in _p[_ko].keys():
                    _k = _ko + '_' + _ki

                    # _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

                    # TODO ---- this is kind of messy.  makes plotting GDM easier but isn't general....
                    if ('var' in _k) and ('bias' in _k):
                        _p_flat[_k + '_(EXP)'] = dc(onp.array(np.exp(_p[_ko][_ki])))
                    else:
                        _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

            _param_hist[1].append(_p_flat)
        else:
            _param_hist[1].append(None)
    except:
        _param_hist[1].append(None)
        failed_str += 'Proposal, '

    # TILT.
    try:
        if _cur_params[2] is not None:
            _p = _cur_params[2]['params']._dict
            _p_flat = {}
            for _ko in _p.keys():
                for _ki in _p[_ko].keys():
                    _k = _ko + '_' + _ki

                    # _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

                    # TODO ---- this is kind of messy.  makes plotting GDM easier but isn't general....
                    if ('var' in _k) and ('bias' in _k):
                        _p_flat[_k + '_(EXP)'] = dc(onp.array(np.exp(_p[_ko][_ki])))
                    else:
                        _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

            _param_hist[2].append(_p_flat)
        else:
            _param_hist[2].append(None)
    except:
        _param_hist[1].append(None)
        failed_str += 'Tilt, '

    # ENCODER.
    try:
        if _cur_params[3] is not None:
            _p = _cur_params[3]['params']._dict
            _p_flat = {}
            for _ko in _p.keys():
                for _ki in _p[_ko].keys():
                    _k = _ko + '_' + _ki
                    _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))

            _param_hist[3].append(_p_flat)
        else:
            _param_hist[3].append(None)
    except:
        _param_hist[3].append(None)
        failed_str += 'Encoder, '

    if len(failed_str) > 0:
        print('[WARNING]: Failed logging parameter values: {}.'.format(failed_str))

    return _param_hist


def initial_validation(env, key, true_model, dataset, masks, true_states, opt, do_fivo_sweep_jitted, _smc_jit, smc_kwargs,
                       num_particles=1000, dset_to_plot=0, init_model=None, GLOBAL_PLOT=True, do_print=None, do_plot=True):
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
    true_lml, em_log_marginal_likelihood, true_lml, initial_fivo_bound, initial_lml = 0.0, 0.0, 0.0, 0.0, 0.0
    init_bpf_posterior, em_posterior, true_bpf_posterior, init_smc_posterior, em_posterior, sweep_fig, filt_fig = None, None, None, None, None, None, None
    em_log_marginal_likelihood = np.nan

    # Test against EM (which for the LDS is exact).
    if hasattr(true_model, 'e_step'):
        # Also normalize by the length of the trace.
        em_posterior = jax.vmap(true_model.e_step)(dataset)
        em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior) / dataset.shape[1]
        em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    true_bpf_posterior = _smc_jit(subkey, true_model, dataset, masks, num_particles=num_particles, **smc_kwargs)
    true_neg_fivo_bound = - np.mean(true_bpf_posterior.log_normalizer)
    true_neg_lml = - utils.lexp(true_bpf_posterior.log_normalizer)

    # if init_model is not None:
    #     # Test BPF in the initial model..
    #     key, subkey = jr.split(key)
    #     init_bpf_posterior = _smc_jit(subkey, init_model, dataset, masks, num_particles=num_particles, **smc_kwargs)
    #     initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)

    # Test SMC in the initial model.
    key, subkey = jr.split(key)
    initial_fivo_bound, init_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                                  get_params_from_opt(opt),
                                                                  _num_particles=num_particles,
                                                                  _datasets=dataset,
                                                                  _masks=masks)
    initial_lml = -utils.lexp(init_smc_posterior.log_normalizer)

    # # Dump any odd and ends of test code in here.
    # temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
    #                      num_particles=10, dset_to_plot=dset_to_plot, init_model=init_model)

    # # Do some plotting.
    # if do_plot:
    #
    #     if em_posterior is not None:
    #         sweep_em_mean = em_posterior.mean()[dset_to_plot]
    #         sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
    #         sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    #         _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
    #                            tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])
    #
    #     if true_bpf_posterior is not None:
    #         _plot_single_sweep(true_bpf_posterior[dset_to_plot].filtering_particles,
    #                            true_states[dset_to_plot],
    #                            tag='True BPF Filtering (' + str(num_particles) + ' particles).',
    #                            obs=dataset[dset_to_plot])
    #         _plot_single_sweep(true_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
    #                            true_states[dset_to_plot],
    #                            tag='True BPF Smoothing (' + str(num_particles) + ' particles).',
    #                            obs=dataset[dset_to_plot])
    #
    #     if init_bpf_posterior is not None:
    #         _plot_single_sweep(init_bpf_posterior[dset_to_plot].filtering_particles,
    #                            true_states[dset_to_plot],
    #                            tag='Initial BPF Filtering (' + str(num_particles) + ' particles).',
    #                            obs=dataset[dset_to_plot])
    #         _plot_single_sweep(init_bpf_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
    #                            true_states[dset_to_plot],
    #                            tag='Initial BPF Smoothing (' + str(num_particles) + ' particles).',
    #                            obs=dataset[dset_to_plot])
    #
    #     if init_smc_posterior is not None:
    #         filt_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].filtering_particles,
    #                                       true_states[dset_to_plot],
    #                                       tag='Initial SMC Filtering (' + str(num_particles) + ' particles).',
    #                                       obs=dataset[dset_to_plot])
    #         sweep_fig = _plot_single_sweep(init_smc_posterior[dset_to_plot].sample(sample_shape=(num_particles,), seed=jr.PRNGKey(0)),
    #                                        true_states[dset_to_plot],
    #                                        tag='Initial SMC Smoothing (' + str(num_particles) + ' particles).',
    #                                        obs=dataset[dset_to_plot])
    #     else:
    #         sweep_fig = None
    #         filt_fig = None
    # else:
    #     sweep_fig = None
    #     filt_fig = None

    # Do some print.
    if do_print is not None:
        _smoothed_training_loss = initial_lml
        do_print(0, true_model, opt, true_lml, true_neg_fivo_bound, initial_lml, initial_fivo_bound, em_log_marginal_likelihood, _smoothed_training_loss)

    return true_neg_lml, true_neg_fivo_bound, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def test_small_sweeps(key, params, single_fivo_eval_small_vmap, single_bpf_true_eval_small_vmap, em_neg_lml, model=None):
    """
    TODO - this is quite a slow subroutine.  50 reps may be too many.
    Args:
        key:
        params:
        single_fivo_eval_small_vmap:
        single_bpf_true_eval_small_vmap:
        em_neg_lml:
        model:

    Returns:

    """
    n_reps = 50
    small_true_bpf_neg_lml, small_true_bpf_neg_lml_var, small_true_bpf_neg_fivo_mean, small_true_bpf_neg_fivo_var = None, None, None, None
    key, subkey1, subkey2 = jr.split(key, num=3)

    if model != 'VRNN':
        small_true_bpf_posteriors = single_bpf_true_eval_small_vmap(jr.split(subkey1, n_reps))
        small_true_bpf_lml_all = small_true_bpf_posteriors.log_normalizer
        small_true_bpf_neg_lml = - utils.lexp(utils.lexp(small_true_bpf_lml_all, axis=1))
        small_true_bpf_neg_lml_var = np.mean(np.var(small_true_bpf_lml_all, axis=0))
        small_true_bpf_neg_fivo_mean = - np.mean(small_true_bpf_lml_all)
        small_true_bpf_neg_fivo_var = np.var(np.mean(small_true_bpf_lml_all, axis=1))

    small_pred_smc_posteriors = single_fivo_eval_small_vmap(jr.split(subkey2, n_reps), params)
    small_pred_smc_lml_all = small_pred_smc_posteriors.log_normalizer
    small_pred_smc_neg_lml = - utils.lexp(utils.lexp(small_pred_smc_lml_all, axis=1))
    small_pred_smc_neg_lml_var = np.mean(np.var(small_pred_smc_lml_all, axis=0))
    small_pred_smc_neg_fivo_mean = - np.mean(small_pred_smc_lml_all)
    small_pred_smc_neg_fivo_var = np.var(np.mean(small_pred_smc_lml_all, axis=1))

    try:
        small_nlml_metrics = {'mean':       {'em_true':     em_neg_lml,
                                             'bpf_true':    small_true_bpf_neg_lml,
                                             'pred':        small_pred_smc_neg_lml, },
                              'variance':   {'bpf_true':    small_true_bpf_neg_lml_var,
                                             'pred':        small_pred_smc_neg_lml_var}, }
    except:
        small_nlml_metrics = None

    try:
        small_fivo_metrics = {'mean':       {'bpf_true':    small_true_bpf_neg_fivo_mean,
                                             'pred':        small_pred_smc_neg_fivo_mean, },
                              'variance':   {'bpf_true':    small_true_bpf_neg_fivo_var,
                                             'pred':        small_pred_smc_neg_fivo_var, }, }
    except:
        small_fivo_metrics = None

    # try:
    #     _small_pred_smc_posteriors = single_fivo_eval_small_vmap(jr.split(subkey1, n_reps), [utils.mutate_named_tuple_by_key(params[0], {'dynamics_weights': ((params[0][0] * 0.0) + 1.0)}), None, None, None])
    #     _small_pred_smc_lml_all = _small_pred_smc_posteriors.log_normalizer
    #     _small_pred_smc_neg_lml = - utils.lexp(utils.lexp(_small_pred_smc_lml_all, axis=1))
    #     _small_pred_smc_neg_lml_var = np.mean(np.var(_small_pred_smc_lml_all, axis=0))
    #     _small_pred_smc_neg_fivo_mean = - np.mean(_small_pred_smc_lml_all)
    #     _small_pred_smc_neg_fivo_var = np.var(np.mean(_small_pred_smc_lml_all, axis=1))
    #     tmp = ((small_pred_smc_neg_lml, _small_pred_smc_neg_lml, small_true_bpf_neg_lml),
    #             (small_pred_smc_neg_lml_var, _small_pred_smc_neg_lml_var, small_true_bpf_neg_lml_var),
    #             (small_pred_smc_neg_fivo_mean, _small_pred_smc_neg_fivo_mean, small_true_bpf_neg_fivo_mean),
    #             (small_pred_smc_neg_fivo_var, _small_pred_smc_neg_fivo_var, small_true_bpf_neg_fivo_var))
    # except:
    #     tmp = ((small_pred_smc_neg_lml, small_true_bpf_neg_lml),
    #            (small_pred_smc_neg_lml_var, small_true_bpf_neg_lml_var),
    #            (small_pred_smc_neg_fivo_mean, small_true_bpf_neg_fivo_mean),
    #            (small_pred_smc_neg_fivo_var, small_true_bpf_neg_fivo_var))
    # print(tmp)

    return small_nlml_metrics, small_fivo_metrics


def compare_kls(get_marginals, env, opt, dataset, mask, true_model, key, do_fivo_sweep_jitted, smc_jitted, smc_kw_args, plot=True, true_bpf_kls=None):
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

    def compute_marginal_kls(smoothing_particles):
        """

        E_q [ log Q / P ]

        Args:
            smoothing_particles:

        Returns:

        """

        # To compute the marginals we are just going to fit a Gaussian.
        kl_p_q = []
        for _t in range(smoothing_particles.shape[-2]):
            samples = smoothing_particles.squeeze()[:, :, _t]
            q_mu = np.mean(samples, axis=1)
            q_sd = np.std(samples, axis=1)  #  + eps

            p_mu = marginals.mean()[:, _t]
            p_sd = marginals.stddev()[:, _t]  # + eps

            _kl_p_q = np.log(q_sd / p_sd) + \
                      (((p_sd ** 2) + ((p_mu - q_mu) ** 2)) / (2.0 * (q_sd ** 2))) + \
                      - 0.5

            _kl_p_q = _kl_p_q.at[q_sd < eps].set(np.nan)

            kl_p_q.append(_kl_p_q)

        return np.asarray(kl_p_q)

    # Set some defaults.
    num_particles = env.config.sweep_test_particles
    eps = 1e-6

    # Get the analytic smoothing marginals.
    marginals = get_marginals(true_model, dataset)

    if marginals is None:
        # TODO - make this more reliable somehow.
        # If there was no analytic marginal available.
        return np.asarray([np.inf]), np.asarray([np.inf])

    # Compare the KLs of the smoothing distributions.
    if true_bpf_kls is None:
        key, subkey = jr.split(key)
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, mask, num_particles=num_particles, **smc_kw_args)
        true_bpf_kls = compute_marginal_kls(true_bpf_posterior.weighted_smoothing_particles)

    key, subkey = jr.split(key)
    _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                 get_params_from_opt(opt),
                                                 _num_particles=num_particles,
                                                 _datasets=dataset,
                                                 _masks=mask)
    pred_smc_kls = compute_marginal_kls(pred_smc_posterior.weighted_smoothing_particles)

    if plot and env.config.PLOT:
        fig = plt.figure()

        true_median = np.nanquantile(np.asarray(true_bpf_kls), 0.5, axis=1)
        pred_median = np.nanquantile(np.asarray(pred_smc_kls), 0.5, axis=1)

        true_lq = np.nanquantile(np.asarray(true_bpf_kls), 0.25, axis=1)
        pred_lq = np.nanquantile(np.asarray(pred_smc_kls), 0.25, axis=1)

        true_uq = np.nanquantile(np.asarray(true_bpf_kls), 0.75, axis=1)
        pred_uq = np.nanquantile(np.asarray(pred_smc_kls), 0.75, axis=1)

        plt.plot(true_median, label='True (BPF)', c='r')
        plt.plot(pred_median, label='Pred (FIVO-AUX)', c='b')

        plt.plot(true_lq, c='r', linewidth=0.25)
        plt.plot(pred_lq, c='b', linewidth=0.25)

        plt.plot(true_uq, c='r', linewidth=0.25)
        plt.plot(pred_uq, c='b', linewidth=0.25)

        plt.legend()
        plt.grid(True)
        plt.title('E_sweeps [ KL [ p_true[t] || q_pred[t] ] ] (max ' + str(num_particles) + ' particles).  \n' +
                  'NaN KLs (out of ' + str(dataset.shape[0] * dataset.shape[1]) + ') : ' +
                  ' BPF: ' + str(np.sum(np.isnan(true_bpf_kls))) +
                  ' FIVO: ' + str(np.sum(np.isnan(pred_smc_kls))))
        plt.xlabel('Time, t')
        plt.ylabel('KL_t')
        plt.yscale('log')
        plt.pause(0.001)
        plt.savefig('./figs/kl_diff.pdf')
        plt.close(fig)

    try:
        kl_metrics = {'per_time': {'median':    {'bpf_true':        np.nanquantile(true_bpf_kls, 0.5, axis=1),
                                                 'fivo':            np.nanquantile(pred_smc_kls, 0.5, axis=1), },
                                   'lq':        {'bpf_true':        np.nanquantile(true_bpf_kls, 0.25, axis=1),
                                                 'fivo':            np.nanquantile(pred_smc_kls, 0.25, axis=1), },
                                   'uq':        {'bpf_true':        np.nanquantile(true_bpf_kls, 0.75, axis=1),
                                                 'fivo':            np.nanquantile(pred_smc_kls, 0.75, axis=1), },
                                   'mean':      {'bpf_true':        np.nanmean(true_bpf_kls, axis=1),
                                                 'fivo':            np.nanmean(pred_smc_kls, axis=1), },
                                   'variance':  {'bpf_true':        np.nanvar(true_bpf_kls, axis=1),
                                                 'fivo':            np.nanvar(pred_smc_kls, axis=1), },
                                   'nan':       {'bpf_true':        np.sum(np.isnan(true_bpf_kls), axis=1),
                                                 'fivo':            np.sum(np.isnan(pred_smc_kls), axis=1), }, },
                      'expected': {'bpf_true':  np.nanmean(true_bpf_kls),
                                   'pred':      np.nanmean(pred_smc_kls)}
                      }
    except:
        kl_metrics = None

    return kl_metrics, true_bpf_kls


def compare_unqiue_particle_counts(env, opt, dataset, mask, true_model, key, do_fivo_sweep_jitted, smc_jitted, smc_kwargs, plot=True, true_bpf_upc=None):
    """

    Args:
        env:
        opt:
        dataset:
        true_model:
        key:
        do_fivo_sweep_jitted:
        plot:

    Returns:

    """

    if env.config.model == 'VRNN':
        print('[WARNING]: UPC not implemented yet.')
        return np.zeros((dataset.shape[0], dataset.shape[1])), np.zeros((dataset.shape[0], dataset.shape[1]))

    def calculate_unique_particle_counts(_particles):
        """
        This is a pain to JAX, so just do it in a loop.

        Args:
            _particles:

        Returns:

        """
        unique_particle_counts = []
        for _sweep in _particles:
            _unique_particle_counts_at_t = []
            for _t in range(_sweep.shape[1]):
                _unique_particle_counts_at_t.append(len(np.unique(_sweep[..., _t, :], axis=0, return_counts=True)[1]))
            unique_particle_counts.append(_unique_particle_counts_at_t)
        return np.asarray(unique_particle_counts)

    num_particles = env.config.sweep_test_particles

    # Compare the KLs of the smoothing distributions.
    if true_bpf_upc is None:
        key, subkey = jr.split(key)
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, mask, num_particles=num_particles, **smc_kwargs)
        true_bpf_upc = calculate_unique_particle_counts(true_bpf_posterior.weighted_smoothing_particles)

    key, subkey = jr.split(key)
    _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
                                                 get_params_from_opt(opt),
                                                 _num_particles=num_particles,
                                                 _datasets=dataset,
                                                 _masks=mask)
    pred_smc_upc = calculate_unique_particle_counts(pred_smc_posterior.weighted_smoothing_particles)

    if plot and env.config.PLOT:
        fig = plt.figure()
        plt.plot(np.mean(np.asarray(true_bpf_upc), axis=0), label='True (BPF)')
        plt.plot(np.mean(np.asarray(pred_smc_upc), axis=0), label='Pred (FIVO)')

        plt.legend()
        plt.grid(True)
        plt.title(r'E_{sweeps} [ #unique_particles @ t ] (max ' + str(num_particles) + ' particles).')
        plt.xlabel('Time, t')
        plt.ylabel(r'#unique_particles')
        plt.ylim([0.9, num_particles + 0.1])
        plt.plot([0, len(np.mean(np.asarray(true_bpf_upc), axis=0))-1], [1.0, 1.0], c='k', linestyle=':')
        plt.plot([0, len(np.mean(np.asarray(true_bpf_upc), axis=0))-1], [num_particles, num_particles], c='k', linestyle='-.')
        plt.pause(0.001)
        plt.savefig('./figs/ss_diff.pdf')
        plt.close(fig)

    try:
        upc_metrics = {'per_time': {'median':       {'bpf_true': np.quantile(true_bpf_upc, 0.5, axis=0),
                                                     'fivo': np.quantile(pred_smc_upc, 0.5, axis=0), },
                                    'lq':           {'bpf_true': np.quantile(true_bpf_upc, 0.25, axis=0),
                                                     'fivo': np.quantile(pred_smc_upc, 0.25, axis=0), },
                                    'uq':           {'bpf_true': np.quantile(true_bpf_upc, 0.75, axis=0),
                                                     'fivo': np.quantile(pred_smc_upc, 0.75, axis=0), },
                                    'mean':         {'bpf_true': np.mean(true_bpf_upc, axis=0),
                                                     'fivo': np.mean(pred_smc_upc, axis=0), },
                                    'variance':     {'bpf_true': np.var(true_bpf_upc, axis=0),
                                                     'fivo': np.var(pred_smc_upc, axis=0), }, },
                       'expected': {'bpf_true':     np.mean(true_bpf_upc),
                                    'pred':         np.mean(pred_smc_upc), }
                       }
    except:
        upc_metrics = None

    return upc_metrics, true_bpf_upc


def compare_sweeps(env, opt, dataset, mask, true_model, key, do_fivo_sweep_jitted, smc_jitted, smc_kw_args,
                   tag='', nrep=10, true_states=None, num_particles=None):
    """

    Args:
        env:
        opt:
        dataset:
        mask:
        true_model:
        key:
        do_fivo_sweep_jitted:
        smc_jitted:
        tag:
        nrep:
        true_states:
        num_particles:

    Returns:

    """

    if num_particles is None:
        num_particles = env.config.sweep_test_particles

    # BPF in true model.
    key, subkey = jr.split(key)
    final_val_posterior_bpf_true = smc_jitted(subkey,
                                              true_model,
                                              dataset,
                                              mask,
                                              num_particles=num_particles,
                                              **smc_kw_args
                                              )

    # SMC with tilt.
    key, subkey = jr.split(key)
    _, final_val_posterior_fivo_aux = do_fivo_sweep_jitted(subkey,
                                                           get_params_from_opt(opt),
                                                           _num_particles=num_particles,
                                                           _datasets=dataset,
                                                           _masks=mask)

    # CODE for plotting lineages.
    for _dset_idx in range(nrep):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
        plt.suptitle('Tag: ' + str(tag) + ', ' + str(num_particles) + ' particles.')

        for _i, _p in enumerate(final_val_posterior_bpf_true[_dset_idx].weighted_smoothing_particles):
            for __i, __p in enumerate(_p.T):
                ax[0].plot(__p, linewidth=0.1, c=color_names[__i], label='Smoothing particles (BPF)' if ((__i == 0) and (_i == 0)) else None)
        ax[0].grid(True)
        ax[0].set_title('BPF in true model')

        for _i, _p in enumerate(final_val_posterior_fivo_aux[_dset_idx].weighted_smoothing_particles):
            for __i, __p in enumerate(_p.T):
                ax[1].plot(__p, linewidth=0.1, c=color_names[__i], label='Smoothing particles (FIVO)' if ((__i == 0) and (_i == 0)) else None)
        ax[1].grid(True)

        # # Plot the observed data.
        # for _i, _p in enumerate(dataset[_dset_idx].T):
        #     ax[0].plot(_p, linewidth=1.0, c=color_names[_i], linestyle=':', label='Observed data' if _i == 0 else None)
        #     ax[1].plot(_p, linewidth=1.0, c=color_names[_i], linestyle=':', label='Observed data' if _i == 0 else None)

        if (get_params_from_opt(opt)[1] is not None) and (get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pqr.')
            _tag = 'pqr'
        elif (get_params_from_opt(opt)[1] is not None) and (get_params_from_opt(opt)[2] is None):
            ax[1].set_title('SMC-AUX with learned pq.')
            _tag = 'pq'
        elif (get_params_from_opt(opt)[1] is None) and (get_params_from_opt(opt)[2] is not None):
            ax[1].set_title('SMC-AUX with learned pr.')
            _tag = 'pr'
        else:
            ax[1].set_title('SMC-AUX with learned p...?')
            _tag = 'p'

        if true_states is not None:

            if len(true_states.shape) == 2:
                _true_states = true_states
            else:
                _true_states = true_states[_dset_idx]

            for _i, _p in enumerate(_true_states.T):
                ax[0].plot(_p, linewidth=1.0, c=color_names[_i], linestyle='--', label='True states' if _i == 0 else None)
                ax[1].plot(_p, linewidth=1.0, c=color_names[_i], linestyle='--', label='True states' if _i == 0 else None)

        ax[1].legend()

        plt.pause(0.01)
        plt.savefig('./figs/tmp_sweep_{}_{}.pdf'.format(_tag, _dset_idx))
        plt.close(fig)


def pretrain_encoder(env, key, encoder, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks):
    """
    Pretrain the encoder to predict the next step in the sequence using the next-step prediction error.

    Args:
        env:
        key:
        encoder:
        train_datasets:
        train_dataset_masks:
        validation_datasets:
        validation_dataset_masks:

    Returns:

    """
    print('Pretraining data encoder. ')
    rnn_state_dim = env.config.rnn_state_dim if env.config.rnn_state_dim is not None else env.config.latent_dim

    # wrapped_encoder = nn_util.RnnWithReadoutLayer(train_datasets.shape[-1], rnn_state_dim, encoder)
    #
    # key, subkey = jr.split(key)
    # init_carry = wrapped_encoder.initialize_carry(subkey)
    #
    # key, subkey = jr.split(key)
    # encoder_params = wrapped_encoder.init(subkey, *(init_carry, np.zeros(train_datasets.shape[-1])))
    #
    # # We can pre-train the encoder to fit the data.  Do that here.
    # # Note that when training we sweep over the parameters backward train in reverse.
    # enc_opt_def = optim.Adam(learning_rate=env.config.encoder_pretrain_lr)
    # enc_opt = enc_opt_def.create(encoder_params)

    train_mean = np.mean(train_datasets, axis=(0, 1))
    clipped_train_dataset_means = np.clip(train_mean, 0.001, 0.999)
    clipped_log_odds = np.log(clipped_train_dataset_means) - np.log(1 - clipped_train_dataset_means)
    output_bias_init = lambda *_args: clipped_log_odds

    forward_readout_network = nn.Dense(train_datasets.shape[-1], bias_init=output_bias_init)
    backward_readout_network = nn.Dense(train_datasets.shape[-1], bias_init=output_bias_init)

    key, subkey = jr.split(key)
    init_carry = encoder.initialize_carry(subkey)

    key, subkey1, subkey2, subkey3 = jr.split(key, num=4)
    _encoder_params = encoder.init(subkey1, (init_carry, np.zeros(train_datasets.shape[-1])))
    _forward_readout_params = forward_readout_network.init(subkey2, init_carry[0][1])
    _backward_readout_params = backward_readout_network.init(subkey2, init_carry[1][1])
    full_params = (_encoder_params, _forward_readout_params, _backward_readout_params)

    full_opt_def = optim.Adam(learning_rate=env.config.encoder_pretrain_lr)
    full_opt = full_opt_def.create(full_params)

    # Create the batches
    full_idx = np.arange(len(train_datasets))
    total_samples = env.config.encoder_pretrain_opt_steps * env.config.encoder_pretrain_batch_size
    batch_idx = []
    while len(batch_idx) < total_samples:
        key, subkey = jr.split(key)
        mixed_idx = jr.shuffle(subkey, full_idx)
        batch_idx += mixed_idx
    batch_idx = np.asarray(batch_idx)[:total_samples]
    batch_idx = np.reshape(batch_idx, (env.config.encoder_pretrain_opt_steps, env.config.encoder_pretrain_batch_size))

    def _single_loss(_full_params, _key, _obs, _mask):
        _forward_encoded_states, _backward_encoded_states = encoder.encode(_full_params[0], _key, _obs)

        _forward_decoded_states = forward_readout_network.apply(_full_params[1], _forward_encoded_states)
        _backward_decoded_states = backward_readout_network.apply(_full_params[2], _backward_encoded_states)

        _forward_log_probs = tfd.Independent(tfd.Bernoulli(_forward_decoded_states), reinterpreted_batch_ndims=1).log_prob(_obs)
        _backward_log_probs = tfd.Independent(tfd.Bernoulli(_backward_decoded_states), reinterpreted_batch_ndims=1).log_prob(_obs)

        _forward_loss = - np.sum(_forward_log_probs) / np.sum(_mask)
        _backward_loss = - np.sum(_backward_log_probs) / np.sum(_mask)

        return _forward_loss + _backward_loss

    @jax.jit
    def loss(_key, _full_params, _batch, _batch_mask):
        _subkeys = jr.split(_key, len(_batch))
        _losses = jax.vmap(_single_loss, in_axes=(None, 0, 0, 0))(_full_params, _subkeys, _batch, _batch_mask)
        return np.mean(_losses)

    # Compile into val and grad.
    loss_val_and_grad = jax.value_and_grad(loss, argnums=1)

    print_regularity = 10
    min_loss = np.inf

    for _step in range(env.config.encoder_pretrain_opt_steps):
        key, subkey = jr.split(key)
        batch = train_datasets[batch_idx[_step]]
        batch_mask = train_dataset_masks[batch_idx[_step]]
        loss, grad = loss_val_and_grad(subkey, full_opt.target, batch, batch_mask)
        full_opt = full_opt.apply_gradient(grad)

        if _step % print_regularity == 0: print('{:> 6d}: {:> 7.3f}'.format(_step, loss))
        if loss < min_loss: min_loss = loss

    # Now we return just the parameters of the RNN.
    rnn_params = full_opt.target[0]
    print('Done pretraining data encoder. ')
    return rnn_params


def load_piano_data(dataset_pickle_name, phase='train'):
    """

    Returns:

    """
    from ssm.inference.data.datasets import sparse_pianoroll_to_dense

    with open('./data/' + dataset_pickle_name + '.pkl', 'rb') as f:
        dataset_sparse = pickle.load(f)

    PAD_FLAG = 0.0
    MAX_LENGTH = {'jsb': 10000,
                  'piano-midi': 10000,
                  'musedata': 10000,
                  'nottingham': 300}

    min_note = 21
    max_note = 108
    num_notes = max_note - min_note + 1

    dataset_and_metadata = [sparse_pianoroll_to_dense(_d, min_note=min_note, num_notes=num_notes) for _d in dataset_sparse[phase]]
    max_length = max([_d[1] for _d in dataset_and_metadata])

    if max_length > MAX_LENGTH[dataset_pickle_name]:
        max_length = MAX_LENGTH[dataset_pickle_name]

    dataset_masks = []
    dataset = []
    removed_datasets = 0
    for _i, _d in enumerate(dataset_and_metadata):

        if len(_d[0]) > max_length:
            # print('[WARNING]: Removing dataset, over length {}.'.format(max_length))
            removed_datasets += 1
            continue

        dataset.append(np.concatenate((_d[0], PAD_FLAG * np.ones((max_length - len(_d[0]), *_d[0].shape[1:])))))
        dataset_masks.append(np.concatenate((np.ones(_d[0].shape[0]), 0.0 * np.ones((max_length - len(_d[0]))))))

    dataset = np.asarray(dataset)
    dataset_masks = np.asarray(dataset_masks)
    dataset_means = np.asarray(dataset_sparse['train_mean'])
    true_states = None  # There are no true states!

    if removed_datasets > 0:
        total_datasets = removed_datasets + len(dataset)
        removed_percent = 100.0 * float(removed_datasets) / float(total_datasets)
        print('\n\n[WARNING]:')
        print('[WARNING]: Removed {:5.2f}pct of datasets (all datasets over length {}).'.format(removed_percent, max_length))
        print('[WARNING]:\n\n')

    # print('\n\n[WARNING]: trimming data further. \n\n')
    # dataset = dataset[:, :20]
    # dataset_masks = dataset_masks[:, :20]

    print('{}: Loaded {} datasets.'.format(dataset_pickle_name, len(dataset)))

    return dataset, dataset_masks, true_states, dataset_means


# def final_validation(get_marginals,
#                      env,
#                      opt,
#                      dataset,
#                      mask,
#                      true_model,
#                      rebuild_model_fn,
#                      rebuild_prop_fn,
#                      rebuild_tilt_fn,
#                      key,
#                      do_fivo_sweep_jitted,
#                      smc_jitted,
#                      GLOBAL_PLOT=True,
#                      tag=''):
#     """
#
#     Args:
#         get_marginals:
#         env:
#         opt:
#         dataset:
#         true_model:
#         rebuild_model_fn:
#         rebuild_prop_fn:
#         rebuild_tilt_fn:
#         key:
#         do_fivo_sweep_jitted:
#
#     Returns:
#
#     """
#     compare_sweeps(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, tag=tag)
#     true_bpf_kls, pred_smc_kls = compare_kls(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True)

# def temp_validation_code(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
#                          num_particles=10, dset_to_plot=0, init_model=None):
#     """
#
#     Args:
#         key:
#         true_model:
#         dataset:
#         true_states:
#         opt:
#         do_fivo_sweep_jitted:
#         _smc_jit:
#         num_particles:
#         dset_to_plot:
#         init_model:
#
#     Returns:
#
#     """
#
#     # Do some sweeps.
#     key, subkey = jr.split(key)
#     smc_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
#     key, subkey = jr.split(key)
#     initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, get_params_from_opt(opt),
#                                                                 num_particles=num_particles,
#                                                                 datasets=dataset)
#
#     # CODE for plotting lineages.
#     idx = 7
#     fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)
#     for _p in smc_posterior[idx].weighted_smoothing_particles:
#         ax[0].plot(_p, linewidth=0.1, c='b')
#     ax[0].grid(True)
#     for _p in sweep_posteriors[idx].weighted_smoothing_particles:
#         ax[1].plot(_p, linewidth=0.1, c='b')
#     ax[1].grid(True)
#     plt.pause(0.01)
#
#     # Compare the variances of the LML estimates.
#     # Test BPF in the initial model..
#     val_bpf_lml, val_fivo_lml = [], []
#     for _ in range(20):
#         key, subkey = jr.split(key)
#         true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=num_particles)
#         true_bpf_lml = - utils.lexp(true_bpf_posterior.log_normalizer)
#         val_bpf_lml.append(true_bpf_lml)
#
#     for _ in range(20):
#         key, subkey = jr.split(key)
#         initial_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey, get_params_from_opt(opt),
#                                                                     num_particles=num_particles,
#                                                                     datasets=dataset)
#         initial_lml = -utils.lexp(sweep_posteriors.log_normalizer)
#         val_fivo_lml.append(initial_lml)
#
#     print('Variance: BPF:      ', np.var(np.asarray(val_bpf_lml)))
#     print('Variance: FIVO-AUX: ', np.var(np.asarray(val_fivo_lml)))

# def compare_ess(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_ess=None):
#     """
#
#     Args:
#         get_marginals:
#         env:
#         opt:
#         dataset:
#         true_model:
#         key:
#         do_fivo_sweep_jitted:
#         smc_jitted:
#         plot:
#         true_bpf_ess:
#
#     Returns:
#
#     """
#
#     def compute_ess(smoothing_particles):
#         """
#
#
#
#         Args:
#             get_marginals:
#             true_model:
#             dataset:
#             smoothing_particles:
#
#         Returns:
#
#         """
#
#         # To compute the marginals we are just going to fit a Gaussian.
#         ess = []
#         for _t in range(smoothing_particles.shape[-2]):
#             samples = smoothing_particles.squeeze()[:, :, _t]
#
#             p_mu = marginals.mean()[:, _t]
#             p_sd = marginals.stddev()[:, _t]
#
#             # Evaluate the probability of the particle sets under each marginal.
#             eval = jax.vmap(lambda _mu, _sd, _s: tfd.MultivariateNormalDiag(np.expand_dims(_mu, 0), np.expand_dims(_sd, 0)).prob(np.expand_dims(_s, 1)))
#             weights = eval(p_mu, p_sd, samples)
#
#             _ess = np.square(np.sum(weights, axis=1)) / np.sum(np.square(weights), axis=1)
#
#             ess.append(_ess)
#
#         return np.asarray(ess)
#
#     # Set some defaults.
#     num_particles = env.config.sweep_test_particles
#
#     # Get the analytic smoothing marginals.
#     marginals = get_marginals(true_model, dataset)
#
#     if marginals is None:
#         # TODO - make this more reliable somehow.
#         # If there was no analytic marginal available.
#         return np.asarray([np.inf])
#
#     # Compare the KLs of the smoothing distributions.
#     if true_bpf_ess is None:
#         key, subkey = jr.split(key)
#         true_bpf_posterior = smc_jitted(subkey, true_model, dataset, num_particles=num_particles)
#         true_bpf_ess = compute_ess(true_bpf_posterior.weighted_smoothing_particles)
#
#     key, subkey = jr.split(key)
#     _, pred_smc_posterior = do_fivo_sweep_jitted(subkey,
#                                                  get_params_from_opt(opt),
#                                                  _num_particles=num_particles,
#                                                  _datasets=dataset)
#     pred_smc_ess = compute_ess(pred_smc_posterior.weighted_smoothing_particles)
#
#     if plot and env.config.PLOT:
#         fig = plt.figure()
#         plt.plot(np.mean(np.asarray(true_bpf_ess), axis=1), label='True (BPF)')
#         plt.plot(np.mean(np.asarray(pred_smc_ess), axis=1), label='Pred (FIVO-AUX)')
#         # plt.plot(np.median(np.asarray(init_bpf_kls), axis=1), label='bpf')
#         plt.legend()
#         plt.grid(True)
#         plt.title('E_sweeps [ ess_t ] (max ' + str(num_particles) + ' particles).')
#         plt.xlabel('Time, t')
#         plt.ylabel('ESS_t')
#         plt.ylim([0.9, num_particles + 0.1])
#         plt.plot([0, len(np.mean(np.asarray(true_bpf_ess), axis=1))-1], [1.0, 1.0], c='k', linestyle=':')
#         plt.plot([0, len(np.mean(np.asarray(true_bpf_ess), axis=1))-1], [num_particles, num_particles], c='k', linestyle='-.')
#         plt.pause(0.001)
#         plt.savefig('./figs/ESS_diff.pdf')
#         plt.close(fig)
#
#
#    'expected_ess': {'bpf_true':          np.mean(true_bpf_ess),
#                     'pred':              np.mean(pred_smc_ess), },
#
#    'ess': {'median':     {'bpf_true':    np.quantile(true_bpf_ess, 0.5, axis=0),
#                           'pred':        np.quantile(pred_smc_ess, 0.5, axis=0), },
#            'lq':         {'bpf_true':    np.quantile(true_bpf_ess, 0.25, axis=0),
#                           'pred':        np.quantile(pred_smc_ess, 0.25, axis=0), },
#            'uq':         {'bpf_true':    np.quantile(true_bpf_ess, 0.75, axis=0),
#                           'pred':        np.quantile(pred_smc_ess, 0.75, axis=0), },
#            'mean':       {'bpf_true':    np.mean(true_bpf_ess, axis=0),
#                           'pred':        np.mean(pred_smc_ess, axis=0), },
#            'variance':   {'bpf_true':    np.var(true_bpf_ess, axis=0),
#                           'pred':        np.var(pred_smc_ess, axis=0), },
#            },
#
#     return ess_metrics, pred_smc_ess