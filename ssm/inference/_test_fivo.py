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
import ssm.inference.fivo_vi as fivo_vi
import ssm.inference.fivo_util as fivo_util

# If we are on Mac, assume it is a local run
LOCAL_SYSTEM = (('mac' in platform.platform()) or ('Mac' in platform.platform()))

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# # NOTE - this is really useful, but will break parts of the
# # computation graph if NaNs are used for signalling purposes.
# # NaN debugging stuff.
# from jax.config import config
# config.update("jax_debug_nans", True)

# Disable jit for inspection.
DISABLE_JIT = False

# Import and configure WandB.
try:
    import wandb
    USE_WANDB = True
    USERNAME = 'andrewwarrington'
    PROJECT = 'fivo-aux-gamma'
except:
    USE_WANDB = False
    print('[Soft Warning]:  Couldnt configure WandB.')


def main():

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Set up the experiment and log to WandB
        env, key, do_print, define_test, do_plot, get_marginals = fivo.do_fivo_config(USE_WANDB, PROJECT, USERNAME, LOCAL_SYSTEM)

        # Define some holders that will be overwritten later.
        true_nlml, em_log_marginal_likelihood, pred_em_nlml, true_bpf_nlml = 0.0, 0.0, 0.0, 0.0
        filt_fig, sweep_fig_filter, sweep_fig_smooth, true_bpf_kls, true_bpf_upc, true_bpf_ess = None, None, None, None, None, None,

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey, env)

        # Unpack that big mess of stuff.
        true_model, true_states, train_datasets, train_dataset_masks, validation_datasets, validation_dataset_masks = ret_vals[0]  # Unpack true model
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]    # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]        # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                # Unpack tilt.

        # In here we can re-build models from saves if we want to.
        try:
            if env.config.load_path is not None:
                with open(env.config.load_path, 'rb') as f:
                    loaded_params = p.load(f)
                model = rebuild_model_fn(utils.make_named_tuple(loaded_params[0]))
                proposal_params = loaded_params[1]
                tilt_params = loaded_params[2]
        except:
            pass

        # Decide if we are going to anneal the tilt.
        # This is done by dividing the log tilt value by a temperature.
        if env.config.temper > 0.0:
            temper_param = env.config.temper
            tilt_temperatures = - (1.0 - np.square((temper_param / np.linspace(0.1, temper_param, num=env.config.opt_steps + 1)))) + 1.0
            print('\n\n[WARNING]: USING TILT TEMPERING. \n\n')
        else:
            tilt_temperatures = np.ones(env.config.opt_steps + 1,) * 1.0

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Build up the optimizer.
        opt = fivo.define_optimizer(p_params=get_model_free_params(model),
                                    q_params=proposal_params,
                                    r_params=tilt_params,
                                    lr_p=env.config.lr_p,
                                    lr_q=env.config.lr_q,
                                    lr_r=env.config.lr_r,
                                    )

        # Jit the smc subroutine for completeness.
        smc_jit = jax.jit(smc, static_argnums=(7, 8))

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _key, _params, _num_particles, _datasets, _masks, _temperature=1.0: \
            fivo.do_fivo_sweep(_params,
                               _key,
                               rebuild_model_fn,
                               rebuild_prop_fn,
                               rebuild_tilt_fn,
                               _datasets,
                               _masks,
                               _num_particles,
                               **{'use_stop_gradient_resampling': env.config.use_sgr,
                                  'tilt_temperature': _temperature,
                                  'resampling_criterion': env.config.resampling_criterion})

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=(2, ))

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Test the initial models.
        key, subkey = jr.split(key)
        true_nlml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_nlml, initial_fivo_bound = \
            fivo_util.initial_validation(env,
                                         key,
                                         true_model,
                                         validation_datasets,
                                         validation_dataset_masks,
                                         true_states,
                                         opt,
                                         do_fivo_sweep_jitted,
                                         smc_jit,
                                         num_particles=env.config.validation_particles,
                                         dset_to_plot=env.config.dset_to_plot,
                                         init_model=model,
                                         do_print=do_print,
                                         do_plot=False)  # TODO - re-enable plotting.  env.config.PLOT)

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Wrap another call to fivo for validation.
        @jax.jit
        def _single_bpf_true_eval_small(_subkey):
            _sweep_posteriors = smc_jit(_subkey,
                                        true_model,
                                        validation_datasets,
                                        validation_dataset_masks,
                                        num_particles=env.config.sweep_test_particles,
                                        resampling_criterion=env.config.resampling_criterion)
            return _sweep_posteriors.log_normalizer

        # Wrap another call to fivo for validation.
        @jax.jit
        def _single_fivo_eval_small(_subkey, _params):
            _val_fivo_bound, _sweep_posteriors = do_fivo_sweep_jitted(_subkey,
                                                                      _params,
                                                                      _datasets=validation_datasets,
                                                                      _masks=validation_dataset_masks,
                                                                      _num_particles=env.config.sweep_test_particles,)
            return _sweep_posteriors.log_normalizer, _val_fivo_bound

        single_bpf_true_eval_small_vmap = jax.vmap(_single_bpf_true_eval_small)
        single_fivo_eval_small_vmap = jax.vmap(_single_fivo_eval_small, in_axes=(0, None))

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Set up a buffer for doing an alternate VI loop.
        VI_USE_VI_GRAD = env.config.vi_use_tilt_gradient and (opt[2] is not None)
        if VI_USE_VI_GRAD:
            VI_BUFFER_LENGTH = env.config.vi_buffer_length * env.config.datasets_per_batch * env.config.num_particles
            VI_MINIBATCH_SIZE = env.config.vi_minibatch_size
            VI_EPOCHS = env.config.vi_epochs

            # Use this update frequency to roughly match that in FIVO-AUX.
            VI_FREQUENCY = int(VI_EPOCHS * np.floor(VI_BUFFER_LENGTH / VI_MINIBATCH_SIZE))

            # Define defaults.
            VI_STATE_BUFFER = []
            VI_OBS_BUFFER = []
            VI_MASK_BUFFER = []

            # Jit the update function/
            do_vi_tilt_update_jit = jax.jit(fivo_vi.do_vi_tilt_update, static_argnums=(1, 3, 4, 8, 9))

            print('\nVI HYPERPARAMETERS:')
            print('\tVI_BUFFER_LENGTH:', VI_BUFFER_LENGTH)
            print('\tVI_MINIBATCH_SIZE:', VI_MINIBATCH_SIZE)
            print('\tVI_EPOCHS:', VI_EPOCHS)
            print('\tVI_FREQUENCY:', VI_FREQUENCY)
            print('\tVI_USE_VI_GRAD:', VI_USE_VI_GRAD, '\n')

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Define some storage.
        param_hist = [[], [], []]  # Model, proposal, tilt.
        val_hist = [[], [], []]  # Model, proposal, tilt.
        param_figures = [None, None, None]  # Model, proposal, tilt.
        nlml_hist = []
        smoothed_training_loss = 0.0

        # Back up the true parameters.
        true_hist = fivo_util.log_params([[], [], []], [get_model_free_params(true_model), None, None],)  # Model, proposal, tilt.

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Main training loop.
        for _step in range(1, env.config.opt_steps + 1):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(env.config.datasets_per_batch, ), minval=0, maxval=len(train_datasets))
            batched_dataset = train_datasets.at[idx].get()
            batched_dataset_masks = train_dataset_masks.at[idx].get()
            temperature = tilt_temperatures[_step]

            # Do the sweep and compute the gradient.
            cur_params = dc(fivo.get_params_from_opt(opt))
            key, subkey = jr.split(key)
            (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                 fivo.get_params_from_opt(opt),
                                                                                 env.config.num_particles,
                                                                                 batched_dataset,
                                                                                 batched_dataset_masks,
                                                                                 temperature)

            # ----------------------------------------------------------------------------------------------------------------------------------------

            # Apply the gradient update.
            if np.isfinite(pred_fivo_bound):
                if not VI_USE_VI_GRAD:
                    opt = fivo.apply_gradient(grad, opt, )
                else:
                    # Apply the gradient to the model and proposal
                    _opt = fivo.apply_gradient(grad, tuple((opt[0], opt[1])), )
                    opt = tuple((*_opt, opt[2]))  # Recapitulate the full optimizer.
            else:
                print('[WARNING]: Skipped step: ', _step, min(smc_posteriors.log_normalizer), pred_fivo_bound)

            # If we are using the VI tilt gradient and it is a VI epoch, do that step.
            key, subkey = jr.split(key)
            if VI_USE_VI_GRAD and (opt[2] is not None):

                # Add the most recent sweep to the buffer.
                if (len(VI_STATE_BUFFER) * env.config.datasets_per_batch * env.config.num_particles) >= VI_BUFFER_LENGTH:
                    VI_BUFFER_FULL = True
                    del VI_STATE_BUFFER[0]
                    del VI_OBS_BUFFER[0]
                    del VI_MASK_BUFFER[0]
                else:
                    VI_BUFFER_FULL = False
                VI_STATE_BUFFER.append(smc_posteriors.weighted_particles[0])
                VI_OBS_BUFFER.append(batched_dataset)
                VI_MASK_BUFFER.append(batched_dataset)

                if (_step % VI_FREQUENCY == 0 and VI_BUFFER_FULL) or _step == 1:
                    _vi_opt, final_vi_elbo, vi_gradient_steps = do_vi_tilt_update_jit(subkey,
                                                                                      env,
                                                                                      fivo.get_params_from_opt(opt),
                                                                                      rebuild_model_fn,
                                                                                      rebuild_tilt_fn,
                                                                                      VI_STATE_BUFFER,
                                                                                      VI_OBS_BUFFER,
                                                                                      VI_MASK_BUFFER,
                                                                                      opt[2],
                                                                                      _epochs=VI_EPOCHS,
                                                                                      _sgd_batch_size=VI_MINIBATCH_SIZE)
                    opt = tuple((opt[0], opt[1], _vi_opt))  # Recapitulate the full optimizer.

            # ----------------------------------------------------------------------------------------------------------------------------------------

            # Do some validation and logging stuff.

            # Quickly calculate a smoothed training loss for quick and dirty plotting.
            smoothed_training_loss = 0.1 * pred_fivo_bound + 0.9 * smoothed_training_loss if smoothed_training_loss != np.nan else pred_fivo_bound

            # Do some validation and give some output.
            if (_step % env.config.validation_interval == 0) or (_step == 1):

                # Capture the parameters.
                param_hist = fivo_util.log_params(param_hist, cur_params)

                # Do a FIVO-AUX sweep.
                key, subkey = jr.split(key)
                pred_fivo_bound_to_print, pred_sweep = do_fivo_sweep_jitted(subkey,
                                                                            fivo.get_params_from_opt(opt),
                                                                            _num_particles=env.config.validation_particles,
                                                                            _datasets=validation_datasets,
                                                                            _masks=validation_dataset_masks)
                pred_nlml = - utils.lexp(pred_sweep.log_normalizer)
                nlml_hist.append(dc(pred_nlml))

                # Test the variance of the estimators.
                key, subkey = jr.split(key)
                small_fivo_nlml_all, val_fivo_bound = single_fivo_eval_small_vmap(jr.split(subkey, 20), fivo.get_params_from_opt(opt))
                small_fivo_nlml = - utils.lexp(utils.lexp(small_fivo_nlml_all, _axis=1))
                small_fivo_expected_nlml_var = np.mean(np.var(small_fivo_nlml_all, axis=0))

                # Test BPF in the true model.
                key, subkey = jr.split(key)
                small_true_bpf_nlml_all = single_bpf_true_eval_small_vmap(jr.split(subkey, 20))
                small_true_bpf_nlml = - utils.lexp(utils.lexp(small_true_bpf_nlml_all, _axis=1))
                small_true_bpf_expected_nlml_var = np.mean(np.var(small_true_bpf_nlml_all, axis=0))

                # Test the KLs.
                key, subkey = jr.split(key)
                true_bpf_kls, pred_smc_kls = fivo_util.compare_kls(get_marginals,
                                                                   env,
                                                                   opt,
                                                                   validation_datasets,
                                                                   validation_dataset_masks,
                                                                   true_model,
                                                                   subkey,
                                                                   do_fivo_sweep_jitted,
                                                                   smc_jit,
                                                                   true_bpf_kls=true_bpf_kls)  # Force re-running this.

                # Compare the number of unique particles.
                key, subkey = jr.split(key)
                true_bpf_upc, pred_smc_upc = fivo_util.compare_unqiue_particle_counts(env,
                                                                                      opt,
                                                                                      validation_datasets,
                                                                                      validation_dataset_masks,
                                                                                      true_model,
                                                                                      subkey,
                                                                                      do_fivo_sweep_jitted,
                                                                                      smc_jit,
                                                                                      true_bpf_upc=true_bpf_upc)  # Force re-running this.

                # # Compare the effective sample size.
                # key, subkey = jr.split(key)
                # true_bpf_ess, pred_smc_ess = fivo.compare_ess(get_marginals,
                #                                               env,
                #                                               opt,
                #                                               validation_datasets,
                #                                               true_model,
                #                                               subkey,
                #                                               do_fivo_sweep_jitted,
                #                                               smc_jit,
                #                                               true_bpf_ess=true_bpf_ess)

                # Do some plotting if we are plotting.
                if env.config.PLOT and ((_step % (env.config.plot_interval * env.config.validation_interval) == 0) or (_step == 1)):

                    # # Do some plotting.
                    # sweep_fig_filter = _plot_single_sweep(
                    #     pred_sweep[env.config.dset_to_plot].filtering_particles,
                    #     true_states[env.config.dset_to_plot],
                    #     tag='{} Filtering.'.format(_step),
                    #     fig=sweep_fig_filter,
                    #     obs=datasets[env.config.dset_to_plot])
                    # sweep_fig_smooth = _plot_single_sweep(
                    #     pred_sweep[env.config.dset_to_plot].weighted_smoothing_particles,
                    #     true_states[env.config.dset_to_plot],
                    #     tag='{} Smoothing.'.format(_step),
                    #     fig=sweep_fig_smooth,
                    #     obs=datasets[env.config.dset_to_plot])

                    key, subkey = jr.split(key)
                    fivo_util.compare_sweeps(env, opt, validation_datasets, validation_dataset_masks, true_model, rebuild_model_fn, rebuild_prop_fn,
                                             rebuild_tilt_fn, subkey, do_fivo_sweep_jitted, smc_jit, tag=_step, nrep=2, true_states=true_states)

                    param_figures = do_plot(param_hist,
                                            nlml_hist,
                                            em_log_marginal_likelihood,
                                            true_nlml,
                                            get_model_free_params(true_model),
                                            param_figures)

                # Log the validation step.
                val_hist = fivo_util.log_params(val_hist, cur_params,)

                # Save out to a temporary location.
                if (env.config.save_path is not None) and (env.config.load_path is None):
                    with open(env.config.save_path, 'wb') as f:
                        params_to_dump = fivo.get_params_from_opt(opt)
                        if params_to_dump[0] is not None:
                            params_to_dump[0] = params_to_dump[0]._asdict()
                        p.dump(params_to_dump, f)

                # Dump some stuff out to WandB.
                # NOTE - we don't dump everything here because it hurts WandBs brain.
                try:
                    kl_metrics = {'median':          {'bpf_true':    np.nanquantile(true_bpf_kls, 0.5, axis=1),
                                                      'fivo':        np.nanquantile(pred_smc_kls, 0.5, axis=1), },
                                  'lq':              {'bpf_true':    np.nanquantile(true_bpf_kls, 0.25, axis=1),
                                                      'fivo':        np.nanquantile(pred_smc_kls, 0.25, axis=1), },
                                  'uq':              {'bpf_true':    np.nanquantile(true_bpf_kls, 0.75, axis=1),
                                                      'fivo':        np.nanquantile(pred_smc_kls, 0.75, axis=1), },
                                  'mean':            {'bpf_true':    np.nanmean(true_bpf_kls, axis=1),
                                                      'fivo':        np.nanmean(pred_smc_kls, axis=1), },
                                  'variance':        {'bpf_true':    np.nanvar(true_bpf_kls, axis=1),
                                                      'fivo':        np.nanvar(pred_smc_kls, axis=1), },
                                  'nan':             {'bpf_true':    np.sum(np.isnan(true_bpf_kls), axis=1),
                                                      'fivo':        np.sum(np.isnan(pred_smc_kls), axis=1), },
                                  }
                except:
                    kl_metrics = None

                try:
                    upc_metrics = {'median':     {'bpf_true':    np.quantile(true_bpf_upc, 0.5, axis=0),
                                                  'fivo':        np.quantile(pred_smc_upc, 0.5, axis=0), },
                                   'lq':         {'bpf_true':    np.quantile(true_bpf_upc, 0.25, axis=0),
                                                  'fivo':        np.quantile(pred_smc_upc, 0.25, axis=0), },
                                   'uq':         {'bpf_true':    np.quantile(true_bpf_upc, 0.75, axis=0),
                                                  'fivo':        np.quantile(pred_smc_upc, 0.75, axis=0), },
                                   'mean':       {'bpf_true':    np.mean(true_bpf_upc, axis=0),
                                                  'fivo':        np.mean(pred_smc_upc, axis=0), },
                                   'variance':   {'bpf_true':    np.var(true_bpf_upc, axis=0),
                                                  'fivo':        np.var(pred_smc_upc, axis=0), },
                                  }
                except:
                    upc_metrics = None

                to_log = {'step': _step,
                          'params_p_true': true_hist[0][-1],
                          'params_p_pred': param_hist[0][-1],
                          'params_q_pred': param_hist[1][-1],
                          'params_r_pred': param_hist[2][-1],
                          'true_nlml': true_nlml,
                          'pred_nlml': pred_nlml,
                          'pred_fivo_bound': pred_fivo_bound_to_print,
                          'small_fivo_bound': np.mean(np.asarray(val_fivo_bound)),
                          'smoothed_training_loss': smoothed_training_loss,
                          'tilt_temperature': temperature,

                          'small_nlml': {'mean':         {'em_true': em_log_marginal_likelihood,
                                                          'bpf_true': small_true_bpf_nlml,
                                                          'pred': small_fivo_nlml, },
                                         'variance':     {'bpf_true': small_true_bpf_expected_nlml_var,
                                                          'pred': small_fivo_expected_nlml_var}},

                          'kl': kl_metrics,
                          'upc': upc_metrics,

                          'expected_kl': {'bpf_true':   np.nanmean(true_bpf_kls),
                                          'pred':       np.nanmean(pred_smc_kls)},

                          'expected_upc': {'bpf_true':  np.mean(true_bpf_upc),
                                           'pred':      np.mean(pred_smc_upc), },


                          # 'expected_ess': {'bpf_true':          np.mean(true_bpf_ess),
                          #                  'pred':              np.mean(pred_smc_ess), },
                          #
                          # 'ess': {'median':     {'bpf_true':    np.quantile(true_bpf_ess, 0.5, axis=0),
                          #                        'pred':        np.quantile(pred_smc_ess, 0.5, axis=0), },
                          #         'lq':         {'bpf_true':    np.quantile(true_bpf_ess, 0.25, axis=0),
                          #                        'pred':        np.quantile(pred_smc_ess, 0.25, axis=0), },
                          #         'uq':         {'bpf_true':    np.quantile(true_bpf_ess, 0.75, axis=0),
                          #                        'pred':        np.quantile(pred_smc_ess, 0.75, axis=0), },
                          #         'mean':       {'bpf_true':    np.mean(true_bpf_ess, axis=0),
                          #                        'pred':        np.mean(pred_smc_ess, axis=0), },
                          #         'variance':   {'bpf_true':    np.var(true_bpf_ess, axis=0),
                          #                        'pred':        np.var(pred_smc_ess, axis=0), },
                          #         },
                          }

                # If we are not on the local system, push less frequently (or WandB starts to cry).
                if env.config.use_wandb:
                    utils.log_to_wandb(to_log, _epoch=_step, USE_WANDB=env.config.use_wandb, _commit=False)

                    if not env.config.local_system:
                        if ((_step % (env.config.validation_interval * env.config.log_to_wandb_interval)) == 0) or (_step == 1):
                            utils.log_to_wandb()
                    else:
                        utils.log_to_wandb()

                # Do some printing.
                if VI_USE_VI_GRAD:
                    print("VI: Step {:>5d}:  Final VI NEG-ELBO {:> 8.3f}. Steps per update: {:>5d}.  Update frequency {:>5d}.".
                          format(_step, final_vi_elbo, vi_gradient_steps, VI_FREQUENCY))

                do_print(_step,
                         true_model,
                         opt,
                         true_nlml,
                         pred_nlml,
                         pred_fivo_bound_to_print,
                         em_log_marginal_likelihood)

        # Do some final validation.
        fivo_util.final_validation(get_marginals,
                                   env,
                                   opt,
                                   validation_datasets,
                                   validation_dataset_masks,
                                   true_model,
                                   rebuild_model_fn,
                                   rebuild_prop_fn,
                                   rebuild_tilt_fn,
                                   key,
                                   do_fivo_sweep_jitted,
                                   smc_jit)


if __name__ == '__main__':
    main()
    print('Done')
