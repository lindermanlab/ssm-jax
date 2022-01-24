"""
Wrapper for exploring FIVO..
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import pickle as p
import git
import platform
import argparse

from jax import random as jr
from copy import deepcopy as dc
from timeit import default_timer as dt
from jax.config import config
from types import SimpleNamespace
from typing import Iterable
from pprint import pprint

# Import some ssm stuff.
import ssm.utils as utils
import ssm.inference.fivo as fivo
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.inference.smc import _plot_single_sweep
from ssm.inference.smc import smc
from tensorflow_probability.substrates.jax import distributions as tfd

# If we are on Mac, assume it is a local run
LOCAL_SYSTEM = (('mac' in platform.platform()) or ('Mac' in platform.platform()))

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# NOTE - this is really useful, but will break parts of the
# computation graph if NaNs are used for signalling purposes.
# NaN debugging stuff.
# config.update("jax_debug_nans", True)

# Disable jit for inspection.
DISABLE_JIT = False

DEFAULT_MODEL = 'VRNN'

# Import and configure WandB.
try:
    import wandb
    USE_WANDB = True
    USERNAME = 'andrewwarrington'
    PROJECT = 'fivo-aux-gamma'
except:
    USE_WANDB = False
    print('[Soft Warning]:  Couldnt configure WandB.')

# Start the timer.
_st = dt()
_verbose_clock_print = True
clock = lambda __st, __str: utils.clock(__st, __str, _verbose_clock_print)


def do_config():
    """

    Returns:

    """

    # Quickly hack finding the model and importing the right config.
    import sys
    try:
        model = sys.argv[np.where(np.asarray([_a == '--model' for _a in sys.argv]))[0][0] + 1]
    except:
        model = DEFAULT_MODEL
        print('No model specified, defaulting to: ', model)

    if 'LDS' in model:
        from ssm.inference._test_fivo_lds import get_config
    elif 'GDM' in model:
        from ssm.inference._test_fivo_gdm import get_config
    elif 'SVM' in model:
        from ssm.inference._test_fivo_svm import get_config
    elif 'VRNN' in model:
        from ssm.inference._test_fivo_vrnn import get_config
    else:
        raise NotImplementedError()

    # Go and get the model-specific config.
    config_dict, do_print, define_test, do_plot, get_marginals = get_config()

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    if config_dict['free_parameters'] is None or config_dict['free_parameters'] == '':
        config_dict['free_parameters'] = ()
    else:
        config_dict['free_parameters'] = tuple(config_dict['free_parameters'].replace(' ', '').split(','))

    # Do some type conversions.
    config_dict['use_sgr'] = bool(config_dict['use_sgr'])

    # Get everything.
    if USE_WANDB:
        # Set up WandB
        env = wandb.init(project=PROJECT, entity=USERNAME, group=config_dict['log_group'], config=config_dict)
    else:
        log_group = 'none'
        env = SimpleNamespace(**{'config': SimpleNamespace(**config),
                                 'log_group': log_group})

    # Set up some WandB stuff.
    env.config.wandb_group = env.config.log_group
    env.config.use_wandb = bool(USE_WANDB)
    env.config.wandb_project = PROJECT
    env.config.local_system = LOCAL_SYSTEM

    # Grab some git information.
    git_commit, git_branch, git_is_dirty = 'NoneFound', 'NoneFound', 'NoneFound'
    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit = repo.head.object.hexsha
        git_branch = repo.active_branch
        git_is_dirty = repo.is_dirty()
    except:
        print('Failed to grab git info...')
    finally:
        env.config.git_commit = git_commit
        env.config.git_branch = git_branch
        env.config.git_is_dirty = git_is_dirty

    # Do some final bits.
    if len(env.config.free_parameters) == 0: print('\nWARNING: NO FREE MODEL PARAMETERS...\n')
    pprint(env.config)
    return env, do_print, define_test, do_plot, get_marginals


def main():

    global _st

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Set up the experiment and log to WandB
        env, do_print, define_test, do_plot, get_marginals = do_config()

        # Define some holders that will be overwritten later.
        true_nlml = 0.0
        em_log_marginal_likelihood = 0.0
        pred_em_nlml = 0.0
        filt_fig = None
        sweep_fig_filter = None
        sweep_fig_smooth = None
        true_bpf_nlml = 0.0
        true_bpf_kls = None
        true_bpf_upc = None
        true_bpf_ess = None

        # Set up the first key
        key = jr.PRNGKey(env.config.seed)

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey, env)

        # Unpack that big mess of stuff.
        true_model, true_states, train_dataset, train_dataset_masks, validation_datasets, validation_dataset_masks = ret_vals[0]  # Unpack true model.
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]        # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]            # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                    # Unpack tilt.

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

        smc_fixed_args = {'use_stop_gradient_resampling': env.config.use_sgr,
                          'resampling_criterion': env.config.resampling_criterion, }

        # Close over constant parameters.
        def do_fivo_sweep_closed(_key, _params, _num_particles, _datasets, _masks, _temperature=1.0):
            _smc_args = dc(smc_fixed_args)
            _smc_args['tilt_temperature'] = _temperature

            assert _masks.shape[0] == _datasets.shape[0], "Unequal first dimension."
            assert _masks.shape[1] == _datasets.shape[1], "Unequal first dimension."

            return fivo.do_fivo_sweep(_params,
                                      _key,
                                      rebuild_model_fn,
                                      rebuild_prop_fn,
                                      rebuild_tilt_fn,
                                      _datasets,
                                      _masks,
                                      _num_particles,
                                      **_smc_args
                                      )

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
            fivo.initial_validation(env,
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
                                        resampling_criterion=env.config.resampling_criterion,
                                        tilt_temperature=1.0, )
            return _sweep_posteriors.log_normalizer

        # Wrap another call to fivo for validation.
        @jax.jit
        def _single_fivo_eval_small(_subkey, _params):
            _val_fivo_bound, _sweep_posteriors = do_fivo_sweep_jitted(_subkey,
                                                                      _params,
                                                                      env.config.sweep_test_particles,
                                                                      validation_datasets,
                                                                      validation_dataset_masks,
                                                                      _temperature=1.0, )
            return _sweep_posteriors.log_normalizer, _val_fivo_bound

        single_bpf_true_eval_small_vmap = jax.vmap(_single_bpf_true_eval_small)
        single_fivo_eval_small_vmap = jax.vmap(_single_fivo_eval_small, in_axes=(0, None))

        # Define some storage.
        param_hist = [[], [], []]  # Model, proposal, tilt.
        val_hist = [[], [], []]  # Model, proposal, tilt.
        param_figures = [None, None, None]  # Model, proposal, tilt.
        nlml_hist = []
        smoothed_training_loss = 0.0

        # Back up the true parameters.
        true_hist = [[], [], []]  # Model, proposal, tilt.
        true_hist = fivo.log_params(true_hist,
                                    [get_model_free_params(true_model), None, None],)

        # Decide if we are going to anneal the tilt.
        # This is done by dividing the log tilt value by a temperature.
        if env.config.temper > 0.0:
            temper_param = env.config.temper

            # tilt_temperatures = - (1.0 - (temper_param / np.linspace(0.1, temper_param, num=env.config.opt_steps + 1))) + 1.0

            tilt_temperatures = - (1.0 - np.square((temper_param / np.linspace(0.1, temper_param, num=env.config.opt_steps + 1)))) + 1.0

            print('\n\nCAUTION: USING TILT TEMPERING. \n\n')
        else:
            tilt_temperatures = np.ones(env.config.opt_steps + 1,) * 1.0

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Set up a buffer for doing an alternate VI loop.
        VI_USE_VI_GRAD = env.config.vi_use_tilt_gradient and (opt[2] is not None)
        VI_BUFFER_LENGTH = env.config.vi_buffer_length * env.config.datasets_per_batch * env.config.num_particles
        VI_MINIBATCH_SIZE = env.config.vi_minibatch_size
        VI_EPOCHS = env.config.vi_epochs

        # Use this update frequency to roughly match that in FIVO-AUX.
        VI_FREQUENCY = int(VI_EPOCHS * np.floor(VI_BUFFER_LENGTH / VI_MINIBATCH_SIZE))

        # Define defaults.
        VI_STATE_BUFFER = []
        VI_OBS_BUFFER = []
        VI_MASK_BUFFER = []

        if VI_USE_VI_GRAD:
            print()
            print('VI HYPERPARAMETERS:')
            print('\tVI_BUFFER_LENGTH:', VI_BUFFER_LENGTH)
            print('\tVI_MINIBATCH_SIZE:', VI_MINIBATCH_SIZE)
            print('\tVI_EPOCHS:', VI_EPOCHS)
            print('\tVI_FREQUENCY:', VI_FREQUENCY)
            print('\tVI_USE_VI_GRAD:', VI_USE_VI_GRAD)
            print()

            do_vi_tilt_update_jit = jax.jit(fivo.do_vi_tilt_update, static_argnums=(1, 3, 4, 9, 10))

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Main training loop.
        for _step in range(1, env.config.opt_steps + 1):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(env.config.datasets_per_batch, ), minval=0, maxval=len(train_dataset))
            batched_dataset = train_dataset.at[idx].get()
            batched_dataset_masks = train_dataset_masks.at[idx].get()

            # Pull out the tilt temperature.
            temperature = tilt_temperatures[_step]

            # Do the sweep and compute the gradient.
            cur_params = dc(fivo.get_params_from_opt(opt))
            key, subkey = jr.split(key)


            (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                 fivo.get_params_from_opt(opt),
                                                                                 env.config.num_particles,
                                                                                 batched_dataset,
                                                                                 batched_dataset_masks,
                                                                                 temperature,
                                                                                 )

            # with jax.disable_jit():  # TODO - INSANE BUG where jitting this function causes it to fail...
            #     (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
            #                                                                          fivo.get_params_from_opt(opt),
            #                                                                          env.config.num_particles,
            #                                                                          batched_dataset,
            #                                                                          batched_dataset_masks,
            #                                                                          temperature,
            #                                                                          )


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
                print('Warning: Skipped step: ', _step, min(smc_posteriors.log_normalizer), pred_fivo_bound)

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
                VI_MASK_BUFFER.append(batched_dataset_masks)

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

            # Do some validation and give some output.
            smoothed_training_loss = 0.1 * pred_fivo_bound + 0.9 * smoothed_training_loss if smoothed_training_loss != 0.0 else pred_fivo_bound
            val_interval = 2500
            plot_interval = 2500
            if (_step % val_interval == 0) or (_step == 1):

                # Capture the parameters.
                param_hist = fivo.log_params(param_hist, cur_params)

                # Do a FIVO-AUX sweep.
                key, subkey = jr.split(key)
                # with jax.disable_jit():
                pred_fivo_bound_to_print, pred_sweep = do_fivo_sweep_jitted(subkey,
                                                                            fivo.get_params_from_opt(opt),
                                                                            _num_particles=env.config.validation_particles,
                                                                            _datasets=validation_datasets,
                                                                            _masks=validation_dataset_masks,
                                                                            _temperature=1.0)
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
                true_bpf_kls, pred_smc_kls = fivo.compare_kls(get_marginals,
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
                true_bpf_upc, pred_smc_upc = fivo.compare_unqiue_particle_counts(env,
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
                if env.config.PLOT and ((_step % plot_interval == 0) or (_step == 1)):

                    # # Do some plotting.
                    # sweep_fig_filter = _plot_single_sweep(
                    #     pred_sweep[env.config.dset_to_plot].filtering_particles,
                    #     true_states[env.config.dset_to_plot],
                    #     tag='{} Filtering.'.format(_step),
                    #     fig=sweep_fig_filter,
                    #     obs=dataset[env.config.dset_to_plot])
                    # sweep_fig_smooth = _plot_single_sweep(
                    #     pred_sweep[env.config.dset_to_plot].weighted_smoothing_particles,
                    #     true_states[env.config.dset_to_plot],
                    #     tag='{} Smoothing.'.format(_step),
                    #     fig=sweep_fig_smooth,
                    #     obs=dataset[env.config.dset_to_plot])

                    key, subkey = jr.split(key)
                    fivo.compare_sweeps(env, opt, validation_datasets, validation_dataset_masks, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, subkey,
                                        do_fivo_sweep_jitted, smc_jit, tag=_step, nrep=2, true_states=true_states)

                    param_figures = do_plot(param_hist,
                                            nlml_hist,
                                            em_log_marginal_likelihood,
                                            true_nlml,
                                            get_model_free_params(true_model),
                                            param_figures)

                # Log the validation step.
                val_hist = fivo.log_params(val_hist,
                                           cur_params,)

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
                utils.log_to_wandb(to_log, _epoch=_step, USE_WANDB=env.config.use_wandb, _commit=False)

                # If we are not on the local system, push less frequently (or WandB starts to cry).
                if not env.config.local_system:
                    if (_step % 10000 == 0) or (_step == 1):
                        utils.log_to_wandb()
                else:
                    utils.log_to_wandb()

                # Do some printing.
                try:
                    if VI_USE_VI_GRAD:
                        print("VI: Step {:>5d}:  Final VI NEG-ELBO {:> 8.3f}. Steps per update: {:>5d}.  Update frequency {:>5d}.".
                              format(_step, final_vi_elbo, vi_gradient_steps, VI_FREQUENCY))
                except:
                    pass
                do_print(_step,
                         true_model,
                         opt,
                         true_nlml,
                         pred_nlml,
                         pred_fivo_bound_to_print,
                         em_log_marginal_likelihood)

        # Do some final validation.
        fivo.final_validation(get_marginals,
                              env,
                              opt,
                              validation_datasets,
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
