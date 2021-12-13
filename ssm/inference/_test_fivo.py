"""
Wrapper for exploring FIVO..
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
from copy import deepcopy as dc
from timeit import default_timer as dt
import numpy as onp

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.inference.smc import _plot_single_sweep
from ssm.inference.smc import smc
import ssm.utils as utils
import ssm.inference.fivo as fivo

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# # NOTE - this is really useful, but will break parts of the
# # computation graph if NaNs are used for signalling purposes.
# # NaN debugging stuff.
# from jax.config import config
# config.update("jax_debug_nans", True)

DISABLE_JIT = False

# from ssm.inference._test_fivo_lds import lds_do_print as do_print
# from ssm.inference._test_fivo_lds import lds_define_test as define_test
# from ssm.inference._test_fivo_lds import lds_do_plot as do_plot

from ssm.inference._test_fivo_gdm import gdm_do_print as do_print
from ssm.inference._test_fivo_gdm import gdm_define_test as define_test
from ssm.inference._test_fivo_gdm import gdm_do_plot as do_plot

# Uncomment this remove the functionality of the plotting code.
_plot_single_sweep = lambda *args, **kwargs: None

# Start the timer.
_st = dt()
_verbose_clock_print = False
clock = lambda __st, __str: utils.clock(__st, __str, _verbose_clock_print)


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _smc_jit,
                       _num_particles=1000, _dset_to_plot=0, _init_model=None):
    """
    Do an test of the true model and the initialized model.
    :param key:
    :param true_model:
    :param dataset:
    :param true_states:
    :param opt:
    :param _do_fivo_sweep_jitted:
    :param _num_particles:
    :param _dset_to_plot:
    :return:
    """
    global _st
    true_lml, em_log_marginal_likelihood = 0.0, 0.0

    _st = clock(_st, 'tmp')

    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.e_step)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)
    sweep_em_mean = em_posterior.mean()[_dset_to_plot]
    sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[_dset_to_plot]
    sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    _plot_single_sweep(sweep_em_statistics, true_states[_dset_to_plot],
                       tag='EM smoothing', preprocessed=True, _obs=dataset[_dset_to_plot])

    _st = clock(_st, 'em')

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    smc_posterior = _smc_jit(subkey, true_model, dataset, num_particles=_num_particles)
    true_lml = - utils.lexp(smc_posterior.log_normalizer)
    _plot_single_sweep(smc_posterior[_dset_to_plot].filtering_particles,
                       true_states[_dset_to_plot],
                       tag='True BPF Filtering.',
                       _obs=dataset[_dset_to_plot])
    key, subkey = jr.split(key)
    _plot_single_sweep(smc_posterior[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=subkey),
                       true_states[_dset_to_plot],
                       tag='True BPF Smoothing.',
                       _obs=dataset[_dset_to_plot])

    _st = clock(_st, 'bpf true')

    if _init_model is not None:
        # Test BPF in the initial model..
        key, subkey = jr.split(key)
        init_bpf_posterior = _smc_jit(subkey, _init_model, dataset, num_particles=_num_particles)
        initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)
        _plot_single_sweep(init_bpf_posterior[_dset_to_plot].filtering_particles,
                           true_states[_dset_to_plot],
                           tag='Initial BPF Filtering.',
                           _obs=dataset[_dset_to_plot])
        key, subkey = jr.split(key)
        _plot_single_sweep(init_bpf_posterior[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=subkey),
                           true_states[_dset_to_plot],
                           tag='Initial BPF Smoothing.',
                           _obs=dataset[_dset_to_plot])
        print('Initial BPF LML: ', initial_bpf_lml)
        _st = clock(_st, 'bpf init')

    # Test SMC in the initial model.
    initial_params = dc(fivo.get_params_from_opt(opt))
    key, subkey = jr.split(key)
    initial_fivo_bound, sweep_posteriors = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                 _num_particles=_num_particles,
                                                                 _datasets=dataset)
    initial_lml = -utils.lexp(sweep_posteriors.log_normalizer)

    filt_fig = _plot_single_sweep(sweep_posteriors[_dset_to_plot].filtering_particles,
                                  true_states[_dset_to_plot],
                                  tag='Initial SMC Filtering.',
                                  _obs=dataset[_dset_to_plot])
    key, subkey = jr.split(key)
    sweep_fig = _plot_single_sweep(sweep_posteriors[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=subkey),
                                   true_states[_dset_to_plot],
                                   tag='Initial SMC Smoothing.',
                                   _obs=dataset[_dset_to_plot])
    _st = clock(_st, 'smc init')

    # Do some print.
    do_print(0, true_model, opt, true_lml, initial_lml, initial_fivo_bound, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def log_params(_param_hist, _cur_params, _cur_lml, _cur_fivo, _cur_em, _step):
    """
    Parse the parameters and store them for printing.

    Args:
        _param_hist:
        _cur_params:

    Returns:

    """

    # MODEL.
    if _cur_params[0] is not None:
        _p = _cur_params[0]._asdict()
        _p_flat = {}
        for _k in _p.keys():
            _p_flat[_k] = dc(onp.array(_p[_k].flatten()))
        _param_hist[0].append(_p_flat)
    else:
        _param_hist[0].append(None)

    # _prop_dict = {'head_mean_kernel': _cur_params[1]._dict['params']['head_mean_fn']['kernel'].flatten(),
    #               'head_mean_bias': _cur_params[1]._dict['params']['head_mean_fn']['bias'].flatten(),
    #               'head_var_bias': np.exp(_cur_params[1]._dict['params']['head_log_var_fn']['bias'])}
    # _param_hist[1].append(_prop_dict)
    #
    # _tilt_dict = {'head_mean_kernel': _cur_params[2]._dict['params']['head_mean_fn']['kernel'].flatten(),
    #               'head_mean_bias': _cur_params[2]._dict['params']['head_mean_fn']['bias'].flatten(),
    #               'head_var_bias': np.exp(_cur_params[2]._dict['params']['head_log_var_fn']['bias'])}
    # _param_hist[2].append(_tilt_dict)

    # PROPOSAL.
    if _cur_params[1] is not None:
        _p = _cur_params[1]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki
                # if ('var' in _k) and ('bias' in _k):
                #     _p_flat[_k + '_EXP'] = onp.array(np.exp(_p[_ko][_ki]))
                # else:
                #     _p_flat[_k] = onp.array(_p[_ko][_ki])
                _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))
        _param_hist[1].append(_p_flat)
    else:
        _param_hist[1].append(None)

    # TILT.
    if _cur_params[2] is not None:
        _p = _cur_params[2]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki
                # if ('var' in _k) and ('bias' in _k):
                #     _p_flat[_k + '_EXP'] = onp.array(np.exp(_p[_ko][_ki]))
                # else:
                #     _p_flat[_k] = onp.array(_p[_ko][_ki])
                _p_flat[_k] = dc(onp.array(_p[_ko][_ki]))
        _param_hist[2].append(_p_flat)
    else:
        _param_hist[2].append(None)

    # Add the loss terms.
    _param_hist[3] = dc(_cur_lml)
    _param_hist[4] = dc(_cur_fivo)
    _param_hist[5] = dc(_cur_em)
    _param_hist[6] = dc(_step)

    return _param_hist


def main():

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        global _st
        _st = clock(_st, 'init')

        # Define some defaults.
        key = jr.PRNGKey(1)
        true_lml = 0.0
        em_log_marginal_likelihood = 0.0
        filt_fig = None
        sweep_fig = None
        dset_to_plot = 2
        num_val_datasets = 50
        validation_particles = 1000
        USE_SGR = True

        # Define the parameters to be used during optimization.
        num_particles = 10
        opt_steps = 100000
        datasets_per_batch = 8

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey)

        _st = clock(_st, 'define_test')

        # Unpack that big mess of stuff.
        true_model, true_states, dataset = ret_vals[0]                  # Unpack true model.
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]    # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]        # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                # Unpack tilt.

        # Build up the optimizer.
        opt = fivo.define_optimizer(p_params=get_model_free_params(model),
                                    q_params=proposal_params,
                                    r_params=tilt_params)

        # Jit the smc subroutine for completeness.
        smc_jit = jax.jit(smc, static_argnums=6)

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _key, _params, _num_particles, _datasets: \
            fivo.do_fivo_sweep(_params,
                               _key,
                               rebuild_model_fn,
                               rebuild_prop_fn,
                               rebuild_tilt_fn,
                               _datasets,
                               _num_particles,
                               **{'use_stop_gradient_resampling': USE_SGR})

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=(2, ))

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        _st = clock(_st, 'jitting')

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound = \
            initial_validation(key, true_model, dataset[:num_val_datasets], true_states, opt, do_fivo_sweep_jitted, smc_jit,
                               _num_particles=validation_particles, _dset_to_plot=dset_to_plot, _init_model=model)

        _st = clock(_st, 'init val')

        # Back up the true parameters.
        true_hist = [[], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, step.
        true_hist = log_params(true_hist,
                               [get_model_free_params(true_model), None, None],
                               true_lml,
                               0.0,
                               em_log_marginal_likelihood,
                               0)

        # Define some storage.
        param_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        val_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        lml_hist = []
        param_figures = [None, None, None, None]  # Loss, Model, proposal, tilt.

        # Main training loop.
        for _step in range(1, opt_steps):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(datasets_per_batch, ), minval=0, maxval=len(dataset))
            batched_dataset = dataset.at[idx].get()

            # Do the sweep and compute the gradient.
            _st = clock(_st, 'misc')
            key, subkey = jr.split(key)
            cur_params = dc(fivo.get_params_from_opt(opt))
            (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                 fivo.get_params_from_opt(opt),
                                                                                 num_particles,
                                                                                 batched_dataset)
            pred_lml = - utils.lexp(smc_posteriors.log_normalizer)

            # Apply the gradient update.
            opt = fivo.apply_gradient(grad, opt, )

            # Log.
            _st = clock(_st, 'fivo_step')
            if _step % 10 == 0:
                lml_hist.append(dc(pred_lml))
                param_hist = log_params(param_hist, cur_params, pred_lml, pred_fivo_bound, _step)

            # Do some validation and give some output.
            coldstart = 0
            if (_step % 2500 == 0) or (_step < coldstart):
                pred_fivo_bound_to_print = pred_fivo_bound
                pred_lml_to_print = pred_lml

                if _step > coldstart:
                        key, subkey = jr.split(key)

                        pred_fivo_bound_to_print, pred_sweep = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                                    _num_particles=validation_particles,
                                                                                    _datasets=dataset[:num_val_datasets])
                        pred_lml_to_print = - utils.lexp(pred_sweep.log_normalizer)

                        filt_fig = _plot_single_sweep(pred_sweep[dset_to_plot].filtering_particles,
                                                      true_states[dset_to_plot],
                                                      tag='{} Filtering.'.format(_step),
                                                      fig=filt_fig,
                                                      _obs=dataset[dset_to_plot])

                        sweep_fig = _plot_single_sweep(
                            pred_sweep[dset_to_plot]._smoothing_particles,
                            true_states[dset_to_plot],
                            tag='{} Smoothing.'.format(_step),
                            fig=sweep_fig,
                            _obs=dataset[dset_to_plot])

                        pred_em_posterior = jax.vmap(true_model.e_step)(dataset)
                        pred_em_lml = true_model.marginal_likelihood(dataset, posterior=pred_em_posterior)
                        pred_em_lml = - utils.lexp(pred_em_lml)

                        # sweep_fig = _plot_single_sweep(
                        #     pred_sweep[dset_to_plot].sample(sample_shape=(num_particles,), seed=subkey),
                        #     true_states[dset_to_plot],
                        #     tag='{} Smoothing.'.format(_step),
                        #     fig=sweep_fig,
                        #     _obs=dataset[dset_to_plot])

                        do_print(_step,
                                 true_model,
                                 opt,
                                 true_lml,
                                 pred_lml_to_print,
                                 pred_fivo_bound_to_print,
                                 em_log_marginal_likelihood)
                        param_figures = do_plot(param_hist,
                                                lml_hist,
                                                em_log_marginal_likelihood,
                                                true_lml,
                                                get_model_free_params(true_model),
                                                param_figures)

                        # Log the validation step.
                        val_hist = log_params(val_hist,
                                              cur_params,
                                              pred_lml_to_print,
                                              pred_fivo_bound_to_print,
                                              pred_em_lml,
                                              _step)

                # # NOTE - Uncomment this to enable more output for debugging.
                # else:
                #     do_print(_step,
                #              true_model,
                #              opt,
                #              true_lml,
                #              pred_lml_to_print,
                #              pred_fivo_bound_to_print,
                #              em_log_marginal_likelihood)


if __name__ == '__main__':
    main()
    print('Done')
