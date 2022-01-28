import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax import random as jr
from copy import deepcopy as dc

# Import some ssm stuff.
import ssm.utils as utils
from ssm.inference.smc import _plot_single_sweep
from ssm.inference.fivo import get_params_from_opt
from ssm.utils import Verbosity

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
        print('[WARNING]: Failed logging parameter values: model.')
        _param_hist[0].append(None)

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
        print('[WARNING]: Failed logging parameter values: proposal.')
        _param_hist[1].append(None)

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
        print('[WARNING]: Failed logging parameter values: tilt.')
        _param_hist[1].append(None)

    return _param_hist


def initial_validation(env, key, true_model, dataset, masks, true_states, opt, do_fivo_sweep_jitted, _smc_jit,
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
    true_lml, em_log_marginal_likelihood = 0.0, 0.0
    init_bpf_posterior = None
    em_posterior = None
    true_bpf_posterior = None
    true_lml = 0.0
    initial_fivo_bound = 0.0
    init_smc_posterior = None
    initial_lml = 0.0
    em_posterior = None
    em_log_marginal_likelihood = np.nan

    # Test against EM (which for the LDS is exact).
    if hasattr(true_model, 'e_step'):
        em_posterior = jax.vmap(true_model.e_step)(dataset)
        em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
        em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    true_bpf_posterior = _smc_jit(subkey, true_model, dataset, masks, num_particles=num_particles, resampling_criterion=env.config.resampling_criterion)
    true_lml = - utils.lexp(true_bpf_posterior.log_normalizer)

    if init_model is not None:
        # Test BPF in the initial model..
        key, subkey = jr.split(key)
        init_bpf_posterior = _smc_jit(subkey, init_model, dataset, masks, num_particles=num_particles)
        initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)

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

    # TODO - remove this block.  just forcing some EM plotting.
    if em_posterior is not None:
        sweep_em_mean = em_posterior.mean()[dset_to_plot]
        sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
        sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
        _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
                           tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])

    # Do some plotting.
    if do_plot:
        # if em_posterior is not None:
        #     sweep_em_mean = em_posterior.mean()[dset_to_plot]
        #     sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset_to_plot]
        #     sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
        #     _plot_single_sweep(sweep_em_statistics, true_states[dset_to_plot],
        #                        tag='EM smoothing', preprocessed=True, obs=dataset[dset_to_plot])

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
        else:
            sweep_fig = None
            filt_fig = None
    else:
        sweep_fig = None
        filt_fig = None

    # Do some print.
    if do_print is not None:
        do_print(0, true_model, opt, true_lml, initial_lml, initial_fivo_bound, em_log_marginal_likelihood)

    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def compare_kls(get_marginals, env, opt, dataset, mask, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_kls=None):
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
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, mask, num_particles=num_particles)
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

    return true_bpf_kls, pred_smc_kls


def compare_unqiue_particle_counts(env, opt, dataset, mask, true_model, key, do_fivo_sweep_jitted, smc_jitted, plot=True, true_bpf_upc=None):
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
        true_bpf_posterior = smc_jitted(subkey, true_model, dataset, mask, num_particles=num_particles)
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

    return true_bpf_upc, pred_smc_upc


def compare_sweeps(env, opt, dataset, mask, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key, do_fivo_sweep_jitted, smc_jitted,
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
    # _prop = rebuild_prop_fn(get_params_from_opt(opt)[1])
    # if _prop is not None:
    #     initial_distribution = lambda _dset, _model:  _prop(_dset, _model, np.zeros(dataset.shape[-1], ), 0, _model.initial_distribution(), None)
    # else:
    #     initial_distribution = None

    # BPF in true model.
    key, subkey = jr.split(key)
    final_val_posterior_bpf_true = smc_jitted(subkey,
                                              true_model,
                                              dataset,
                                              mask,
                                              num_particles=num_particles)

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


def final_validation(get_marginals,
                     env,
                     opt,
                     dataset,
                     mask,
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
    compare_sweeps(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted, tag=tag)

    # Compare the KLs.
    true_bpf_kls, pred_smc_kls = compare_kls(get_marginals, env, opt, dataset, true_model, key, do_fivo_sweep_jitted, smc_jitted,
                                             plot=True)


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
#     return true_bpf_ess, pred_smc_ess