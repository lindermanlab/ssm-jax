"""
Wrapper for exploring FIVO..
"""

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
from copy import deepcopy as dc
import pickle as p
import platform
from timeit import default_timer as dt

# Import some ssm stuff.
from ssm.utils import Verbosity, possibly_disable_jit
from ssm.inference.smc import smc, multinomial_resampling, systematic_resampling, always_resample_criterion, never_resample_criterion, ess_criterion
import ssm.utils as utils
import ssm.inference.fivo as fivo
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
# If we are on the remote, then hard disable this.
if not LOCAL_SYSTEM:
    DISABLE_JIT = False

# Set the default model for local debugging.
DEFAULT_MODEL = 'LDS'

# Import and configure WandB.
try:
    import wandb
    USE_WANDB = True
    USERNAME = 'andrewwarrington'
    PROJECT = 'fivo-aux-epsilon'
except:
    USE_WANDB = False
    print('[Soft Warning]:  Couldnt configure WandB.')


def main():

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Set up the experiment and log to WandB
        env, key, do_print, define_test, do_plot, get_marginals = fivo.do_fivo_config(DEFAULT_MODEL, USE_WANDB, PROJECT, USERNAME, LOCAL_SYSTEM)

        # TODO - this is a bit grotty.  type convert the resampling function.
        if env.config.resampling_function == 'multinomial_resampling':
            resampling_function = multinomial_resampling
        elif env.config.resampling_function == 'systematic_resampling':
            resampling_function = systematic_resampling
        else:
            raise NotImplementedError()

        if env.config.train_resampling_criterion == 'always_resample':
            train_resampling_criterion = always_resample_criterion
        elif env.config.train_resampling_criterion == 'never_resample':
            train_resampling_criterion = never_resample_criterion
        elif env.config.train_resampling_criterion == 'ess_criterion':
            train_resampling_criterion = ess_criterion
        else:
            raise NotImplementedError()

        if env.config.eval_resampling_criterion == 'always_resample':
            eval_resampling_criterion = always_resample_criterion
        elif env.config.eval_resampling_criterion == 'never_resample':
            eval_resampling_criterion = never_resample_criterion
        elif env.config.eval_resampling_criterion == 'ess_criterion':
            eval_resampling_criterion = ess_criterion
        else:
            raise NotImplementedError()

        # Define some holders that will be overwritten later.
        large_true_bpf_neg_lml, em_neg_lml, pred_em_nlml, true_bpf_nlml, true_neg_bpf_fivo_bound, initial_smc_neg_fivo_bound = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        filt_fig, sweep_fig_filter, sweep_fig_smooth, true_bpf_kls, true_bpf_upc, true_bpf_ess = None, None, None, None, None, None,
        kl_metrics, upc_metrics = None, None
        param_hist = [[], [], [], []]  # Model, proposal, tilt, encoder.
        param_figures = [None, None, None, None]  # Model, proposal, tilt, encoder.
        nlml_hist = []
        smoothed_training_loss = np.nan
        VI_STATE_BUFFER, VI_OBS_BUFFER, VI_MASK_BUFFER = [], [], []
        VI_BUFFER_FULL = False

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey, env)

        # Unpack that big mess of stuff.
        true_model, true_states, trn_datasets, trn_dataset_masks, val_datasets, val_dataset_masks, tst_datasets, tst_dataset_masks = ret_vals[0]  # Unpack true model
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]    # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]        # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                # Unpack tilt.
        encoder, encoder_params, rebuild_encoder_fn = ret_vals[4]      # Unpack the encoder.

        # In here we can re-build models from saves if we want to.
        try:
            if env.config.load_path is not None:
                with open(env.config.load_path, 'rb') as f:
                    loaded_params = p.load(f)
                model = rebuild_model_fn(utils.make_named_tuple(loaded_params[0]))
                proposal_params = loaded_params[1]
                tilt_params = loaded_params[2]
                encoder_params = loaded_params[3]
        except:
            pass

        # Back up the true parameters.
        true_hist = fivo_util.log_params([[], [], [], []], [get_model_free_params(true_model), None, None, None],)  # Model, proposal, tilt, encoder.

        # Decide if we are going to anneal the tilt.
        # This is done by dividing the log tilt value by a temperature.
        if env.config.temper > 0.0:
            temper_param = env.config.temper
            opt_steps = env.config.opt_steps

            # tilt_temperatures = - (1.0 - np.square((temper_param / np.linspace(0.1, temper_param, num=int(env.config.opt_steps / 2.0) + 1)))) + 1.0
            # tilt_temperatures = np.clip(tilt_temperatures, a_min=1.0)

            # Inverse tempering that is terminated half way through optimization.
            _temps = temper_param * (1.0 / np.linspace(0.05, 1.0, num=opt_steps + 1))
            _temps += 1.0 - _temps[int(opt_steps / 2.0)]
            tilt_temperatures = np.clip(_temps, a_min=1.0, a_max=np.inf)

            print('\n\n[WARNING]: USING TILT TEMPERING. \n\n')
        else:
            tilt_temperatures = np.ones(env.config.opt_steps + 1,) * 1.0

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Build up the optimizer.
        opt = fivo.define_optimizer(p_params=get_model_free_params(model),
                                    q_params=proposal_params,
                                    r_params=tilt_params,
                                    e_params=encoder_params,
                                    lr_p=env.config.lr_p,
                                    lr_q=env.config.lr_q,
                                    lr_r=env.config.lr_r,
                                    lr_e=env.config.lr_e,
                                    )

        def gen_smc_kwargs(_temperature=1.0, _resampling_criteria=eval_resampling_criterion):
            return {'use_stop_gradient_resampling': env.config.use_sgr,
                    'tilt_temperature': _temperature,
                    'resampling_criterion': _resampling_criteria,
                    'resampling_function': resampling_function}

        # Jit the smc subroutine for completeness.
        smc_jit = jax.jit(smc, static_argnums=(7, 8, 9, 11, 12))

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _subkey, _params, _num_particles, _datasets, _masks, _temperature=1.0, _resampling_criterion=eval_resampling_criterion: \
            fivo.do_fivo_sweep(_params,
                               _subkey,
                               rebuild_model_fn,
                               rebuild_prop_fn,
                               rebuild_tilt_fn,
                               rebuild_encoder_fn,
                               _datasets,
                               _masks,
                               _num_particles,
                               env.config.use_bootstrap_initial_distribution,
                               **gen_smc_kwargs(_temperature=_temperature, _resampling_criteria=_resampling_criterion))

        # Jit this badboy.
        do_fivo_sweep_jitted = jax.jit(do_fivo_sweep_closed, static_argnums=(2, 6))

        # Close over a regularized version.
        def do_regularized_fivo_sweep_closed(_subkey, _params, _num_particles, _datasets, _masks, _temperature=1.0, _resampling_criterion=train_resampling_criterion):
            _neg_fivo_bound_unreg, _sweep_posteriors = do_fivo_sweep_jitted(_subkey, _params, _num_particles, _datasets, _masks, _temperature, _resampling_criterion)

            def _l2_loss(x):
                if x is None:
                    return 0.0
                else:
                    return (x ** 2).mean()

            _l2_score = sum([sum([_l2_loss(_w) for _w in jax.tree_leaves(w)]) for w in _params])

            _neg_fivo_bound_reg = _neg_fivo_bound_unreg + (env.config.l2_reg * _l2_score)
            return _neg_fivo_bound_reg, _sweep_posteriors

        # Jit this badboy.
        do_regularized_fivo_sweep_closed_jitted = jax.jit(do_regularized_fivo_sweep_closed, static_argnums=(2, 6))

        # Wrap another call to SMC for validation.
        @jax.jit
        def _single_bpf_true_val_small(_subkey):
            _sweep_posteriors = smc_jit(_subkey,
                                        true_model,
                                        val_datasets,
                                        val_dataset_masks,
                                        num_particles=env.config.sweep_test_particles,
                                        **gen_smc_kwargs(_resampling_criteria=eval_resampling_criterion))
            return _sweep_posteriors

        # Wrap another call to fivo for validation.
        @jax.jit
        def _single_fivo_val_small(_subkey, _params):
            _, _sweep_posteriors = do_fivo_sweep_jitted(_subkey,
                                                        _params,
                                                        _datasets=val_datasets,
                                                        _masks=val_dataset_masks,
                                                        _num_particles=env.config.sweep_test_particles,
                                                        _resampling_criterion=eval_resampling_criterion)
            return _sweep_posteriors

        # Wrap another call to SMC for validation.
        @jax.jit
        def _single_bpf_true_tst_small(_subkey):
            _sweep_posteriors = smc_jit(_subkey,
                                        true_model,
                                        tst_datasets,
                                        tst_dataset_masks,
                                        num_particles=env.config.sweep_test_particles,
                                        **gen_smc_kwargs(_resampling_criteria=eval_resampling_criterion))
            return _sweep_posteriors

        # Wrap another call to fivo for validation.
        @jax.jit
        def _single_fivo_tst_small(_subkey, _params):
            _, _sweep_posteriors = do_fivo_sweep_jitted(_subkey,
                                                        _params,
                                                        _datasets=tst_datasets,
                                                        _masks=tst_dataset_masks,
                                                        _num_particles=env.config.sweep_test_particles,
                                                        _resampling_criterion=eval_resampling_criterion)
            return _sweep_posteriors

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = jax.value_and_grad(do_regularized_fivo_sweep_closed_jitted, argnums=1, has_aux=True)

        single_bpf_true_val_small_vmap = jax.vmap(_single_bpf_true_val_small)
        single_fivo_val_small_vmap = jax.vmap(_single_fivo_val_small, in_axes=(0, None))
        single_bpf_true_tst_small_vmap = jax.vmap(_single_bpf_true_tst_small)
        single_fivo_tst_small_vmap = jax.vmap(_single_fivo_tst_small, in_axes=(0, None))

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

            # Jit the update function/
            do_vi_tilt_update_jit = jax.jit(fivo_vi.do_vi_tilt_update, static_argnums=(1, 3, 4, 5, 10, 11))

            print('\nVI HYPERPARAMETERS:')
            print('\tVI_BUFFER_LENGTH:', VI_BUFFER_LENGTH)
            print('\tVI_MINIBATCH_SIZE:', VI_MINIBATCH_SIZE)
            print('\tVI_EPOCHS:', VI_EPOCHS)
            print('\tVI_FREQUENCY:', VI_FREQUENCY)
            print('\tVI_USE_VI_GRAD:', VI_USE_VI_GRAD, '\n')

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Test the initial models.
        key, subkey = jr.split(key)
        large_true_bpf_neg_lml, true_neg_bpf_fivo_bound, em_neg_lml, sweep_fig, filt_fig, initial_smc_neg_lml, initial_smc_neg_fivo_bound = \
            fivo_util.initial_validation(env,
                                         key,
                                         true_model,
                                         val_datasets,
                                         val_dataset_masks,
                                         true_states,
                                         opt,
                                         do_fivo_sweep_jitted,
                                         smc_jit,
                                         smc_kwargs=gen_smc_kwargs(_temperature=1.0),
                                         num_particles=env.config.validation_particles,
                                         dset_to_plot=env.config.dset_to_plot,
                                         init_model=model,
                                         do_print=do_print,
                                         do_plot=False)  # Re-enable plotting with: env.config.PLOT .

        # We will log the best parameters here.
        best_params_raw = dc(fivo_util.get_params_from_opt(opt))
        best_params_processed = fivo_util.log_params([[], [], [], []], best_params_raw)
        best_val_neg_fivo_bound = dc(initial_smc_neg_fivo_bound)

        # --------------------------------------------------------------------------------------------------------------------------------------------

        # Main training loop.
        st = dt()
        for _step in range(1, env.config.opt_steps + 1):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(env.config.datasets_per_batch, ), minval=0, maxval=len(trn_datasets))
            batched_dataset = trn_datasets.at[idx].get()
            batched_dataset_masks = trn_dataset_masks.at[idx].get()
            temperature = tilt_temperatures[_step]

            # Do the sweep and compute the gradient.
            cur_params = dc(fivo.get_params_from_opt(opt))
            key, subkey = jr.split(key)
            (neg_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                fivo.get_params_from_opt(opt),
                                                                                env.config.num_particles,
                                                                                batched_dataset,
                                                                                batched_dataset_masks,
                                                                                temperature)

            # ----------------------------------------------------------------------------------------------------------------------------------------

            # Apply the gradient update.
            if np.isfinite(neg_fivo_bound):
                if not VI_USE_VI_GRAD:
                    opt = fivo.apply_gradient(grad, opt, )
                else:
                    # Apply the gradient to the model and proposal
                    _opt = fivo.apply_gradient((grad[0], grad[1], None, grad[3]), tuple((opt[0], opt[1], None, opt[3])), )
                    opt = tuple((_opt[0], _opt[1], opt[2], _opt[3]))  # Recapitulate the full optimizer.
            else:
                print('[WARNING]: Skipped step: ', _step, min(smc_posteriors.log_normalizer), neg_fivo_bound)

            # If we are using the VI tilt gradient and it is a VI epoch, do that step.
            key, subkey = jr.split(key)
            if VI_USE_VI_GRAD and (opt[2] is not None):

                # Add the most recent sweep to the buffer.
                if (len(VI_STATE_BUFFER) * env.config.datasets_per_batch * env.config.num_particles) >= VI_BUFFER_LENGTH:
                    VI_BUFFER_FULL = True
                    del VI_STATE_BUFFER[0]
                    del VI_OBS_BUFFER[0]
                    del VI_MASK_BUFFER[0]
                VI_STATE_BUFFER.append(smc_posteriors.weighted_particles[0])
                VI_OBS_BUFFER.append(batched_dataset)
                VI_MASK_BUFFER.append(batched_dataset)

                if (_step % VI_FREQUENCY == 0 and VI_BUFFER_FULL) or _step == 1:
                    _vi_opt, final_vi_elbo, vi_gradient_steps = do_vi_tilt_update_jit(subkey,
                                                                                      env,
                                                                                      fivo.get_params_from_opt(opt),
                                                                                      rebuild_model_fn,
                                                                                      rebuild_tilt_fn,
                                                                                      rebuild_encoder_fn,
                                                                                      VI_STATE_BUFFER,
                                                                                      VI_OBS_BUFFER,
                                                                                      VI_MASK_BUFFER,
                                                                                      opt[2],
                                                                                      _epochs=VI_EPOCHS,
                                                                                      _sgd_batch_size=VI_MINIBATCH_SIZE)
                    opt = tuple((opt[0], opt[1], _vi_opt, opt[3]))  # Recapitulate the full optimizer.

            # ----------------------------------------------------------------------------------------------------------------------------------------

            # Do some validation and logging stuff.

            # Quickly calculate a smoothed training loss for quick and dirty plotting.
            smoothed_training_loss = ((0.1 * neg_fivo_bound + 0.9 * smoothed_training_loss) if np.logical_not(np.isnan(smoothed_training_loss)) else neg_fivo_bound)
        
            # Do some validation and give some output.
            if (_step % env.config.validation_interval == 0) or (_step == 1):

                # Clock the time.
                en = dt()
                sti = dt()

                # Capture the parameters.
                param_hist = fivo_util.log_params(param_hist, cur_params)

                # Do a FIVO-AUX sweep.
                key, subkey = jr.split(key)
                large_pred_smc_neg_fivo_bound, large_pred_sweep = do_fivo_sweep_jitted(subkey,
                                                                                       fivo.get_params_from_opt(opt),
                                                                                       _num_particles=env.config.validation_particles,
                                                                                       _datasets=val_datasets,
                                                                                       _masks=val_dataset_masks)
                large_pred_smc_neg_lml = - utils.lexp(large_pred_sweep.log_normalizer)
                nlml_hist.append(dc(large_pred_smc_neg_lml))
                
                # If we have improved, store these parameters.
                if large_pred_smc_neg_fivo_bound < best_val_neg_fivo_bound:
                    best_params_raw = dc(fivo_util.get_params_from_opt(opt))
                    best_params_processed = fivo_util.log_params([[], [], [], []], best_params_raw)
                    best_val_neg_fivo_bound = dc(large_pred_smc_neg_fivo_bound)

                # Test the variance of the estimators with a small number of particles.
                key, subkey = jr.split(key)
                small_neg_lml_metrics, small_neg_fivo_metrics = fivo_util.test_small_sweeps(subkey,
                                                                                            fivo.get_params_from_opt(opt),
                                                                                            single_fivo_val_small_vmap,
                                                                                            single_bpf_true_val_small_vmap,
                                                                                            em_neg_lml,
                                                                                            model=env.config.model)

                # # Test the KLs.
                # if env.config.model != 'VRNN':
                #     key, subkey = jr.split(key)
                #     kl_metrics, true_bpf_kls = fivo_util.compare_kls(get_marginals,
                #                                                      env,
                #                                                      opt,
                #                                                      val_datasets,
                #                                                      val_dataset_masks,
                #                                                      true_model,
                #                                                      subkey,
                #                                                      do_fivo_sweep_jitted,
                #                                                      smc_jit,
                #                                                      gen_smc_kwargs(),
                #                                                      true_bpf_kls=true_bpf_kls)

                # # Compare the number of unique particles.
                # if env.config.model != 'VRNN':
                #     key, subkey = jr.split(key)
                #     upc_metrics, true_bpf_upc = fivo_util.compare_unqiue_particle_counts(env,
                #                                                                          opt,
                #                                                                          val_datasets,
                #                                                                          val_dataset_masks,
                #                                                                          true_model,
                #                                                                          subkey,
                #                                                                          do_fivo_sweep_jitted,
                #                                                                          smc_jit,
                #                                                                          gen_smc_kwargs(),
                #                                                                          true_bpf_upc=true_bpf_upc)

                # Save out to a temporary location.
                if (env.config.save_path is not None) and (env.config.load_path is None):
                    with open(env.config.save_path, 'wb') as f:
                        params_to_dump = fivo.get_params_from_opt(opt)
                        if params_to_dump[0] is not None:
                            params_to_dump[0] = params_to_dump[0]._asdict()
                        p.dump(params_to_dump, f)

                # Dump some stuff out to WandB.
                # NOTE - we don't dump everything here because it hurts WandBs brain.
                to_log = {'step': _step,
                          'tilt_temperature': temperature,
                          'smoothed_training_loss': smoothed_training_loss,

                          'params': {'p_true': true_hist[0][-1],
                                     'p_pred': param_hist[0][-1],
                                     'q_pred': param_hist[1][-1],
                                     'r_pred': param_hist[2][-1],
                                     'e_pred': param_hist[3][-1], },

                          'best_params': best_params_processed,
                          'best_val_neg_fivo_bound': best_val_neg_fivo_bound,

                          'large_neg_fivo':     {'bpf_true': true_neg_bpf_fivo_bound,
                                                 'pred': large_pred_smc_neg_fivo_bound},
                          'large_neg_lml':      {'em_true': em_neg_lml,
                                                 'bpf_true': large_true_bpf_neg_lml,
                                                 'pred': large_pred_smc_neg_lml},

                          'small_neg_fivo_bound': small_neg_fivo_metrics,
                          'small_neg_lml': small_neg_lml_metrics,

                          'zz_auxiliary_metrics':  {'kl': kl_metrics,
                                                    'upc': upc_metrics, },
                          }

                # If we are not on the local system, push less frequently (or WandB starts to cry).
                if env.config.use_wandb:
                    utils.log_to_wandb(to_log, _epoch=_step, USE_WANDB=env.config.use_wandb, _commit=False)

                    if not env.config.local_system:
                        if ((_step % (env.config.validation_interval * env.config.log_to_wandb_interval)) == 0) or (_step == 1):
                            utils.log_to_wandb()
                    else:
                        utils.log_to_wandb()

                # Do some plotting if we are plotting.
                key, subkey = jr.split(key)
                if env.config.PLOT:

                    # key, subkey = jr.split(key)
                    # fivo_util.compare_sweeps(env, opt, val_datasets, val_dataset_masks, true_model, subkey, do_fivo_sweep_jitted,
                    #                          smc_jit, gen_smc_kwargs(), tag=_step, nrep=2, true_states=true_states)

                    param_figures = do_plot(param_hist, nlml_hist, em_neg_lml, large_true_bpf_neg_lml,
                                            get_model_free_params(true_model), param_figures)

                # Do some printing.
                eni = dt()
                print('Time per gradient step: {:> 6.3f}s'.format((en - st) / env.config.validation_interval))
                print('Time per validation   : {:> 6.3f}s'.format((eni - sti)))
                st = dt()
                if VI_USE_VI_GRAD:
                    print("VI: Step {:>5d}:  Final VI NEG-ELBO {:> 8.3f}. Steps per update: {:>5d}.  Update frequency {:>5d}.".
                          format(_step, final_vi_elbo, vi_gradient_steps, VI_FREQUENCY))
                do_print(_step, true_model, opt, large_true_bpf_neg_lml, true_neg_bpf_fivo_bound, large_pred_smc_neg_lml, large_pred_smc_neg_fivo_bound, em_neg_lml, smoothed_training_loss)

                # # Compare the effective sample size.
                # key, subkey = jr.split(key)
                # ess_metrics, pred_smc_ess = fivo.compare_ess(get_marginals,
                #                                               env,
                #                                               opt,
                #                                               val_datasets,
                #                                               true_model,
                #                                               subkey,
                #                                               do_fivo_sweep_jitted,
                #                                               smc_jit,
                #                                               true_bpf_ess=true_bpf_ess)

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
                
        # Done in the main training loop.
        print("\nDone in the main training loop.")
        print("Applying trained models to the test set.")

        key, subkey = jr.split(key)
        large_best_smc_neg_fivo_bound_val, large_best_sweep_val = do_fivo_sweep_jitted(subkey,
                                                                                       best_params_raw,
                                                                                       _num_particles=env.config.validation_particles,
                                                                                       _datasets=val_datasets,
                                                                                       _masks=val_dataset_masks,
                                                                                       _resampling_criterion=eval_resampling_criterion)
        large_best_smc_neg_lml_val = - utils.lexp(large_best_sweep_val.log_normalizer)

        key, subkey = jr.split(key)
        large_best_smc_neg_fivo_bound_tst, large_best_sweep_tst = do_fivo_sweep_jitted(subkey,
                                                                                       best_params_raw,
                                                                                       _num_particles=env.config.validation_particles,
                                                                                       _datasets=tst_datasets,
                                                                                       _masks=tst_dataset_masks,
                                                                                       _resampling_criterion=eval_resampling_criterion)
        large_best_smc_neg_lml_tst = - utils.lexp(large_best_sweep_tst.log_normalizer)

        # Test the variance of the estimators with a small number of particles.
        key, subkey = jr.split(key)
        small_best_neg_lml_metrics_val, small_best_neg_fivo_metrics_val = fivo_util.test_small_sweeps(subkey,
                                                                                                      best_params_raw,
                                                                                                      single_fivo_val_small_vmap,
                                                                                                      single_bpf_true_val_small_vmap,
                                                                                                      em_neg_lml,
                                                                                                      model=env.config.model)

        # Test the variance of the estimators with a small number of particles.
        key, subkey = jr.split(key)
        small_best_neg_lml_metrics_tst, small_best_neg_fivo_metrics_tst = fivo_util.test_small_sweeps(subkey,
                                                                                                      best_params_raw,
                                                                                                      single_fivo_tst_small_vmap,
                                                                                                      single_bpf_true_tst_small_vmap,
                                                                                                      em_neg_lml,
                                                                                                      model=env.config.model)

        # Compile these into a dictionary for the final log.
        _final_metrics_tst = {'best_large_smc_neg_fivo_bound_tst': large_best_smc_neg_fivo_bound_tst,
                              'best_large_smc_neg_lml_tst': large_best_smc_neg_lml_tst,
                              'best_small_neg_lml_metrics_tst': small_best_neg_lml_metrics_tst,
                              'best_small_neg_fivo_metrics_tst': small_best_neg_fivo_metrics_tst, }
        _final_metrics_val = {'best_large_smc_neg_fivo_bound_val': large_best_smc_neg_fivo_bound_val,
                              'best_large_smc_neg_lml_val': large_best_smc_neg_lml_val,
                              'best_small_neg_lml_metrics_val': small_best_neg_lml_metrics_val,
                              'best_small_neg_fivo_metrics_val': small_best_neg_fivo_metrics_val, }

        final_metrics = {'z_final_metrics_tst': _final_metrics_tst,
                         'z_final_metrics_val': _final_metrics_val,
                         'z_best_params': best_params_processed}
        utils.log_to_wandb(final_metrics, _epoch=_step+1)
        

if __name__ == '__main__':
    print('In main code.')
    main()
    print('Done')
