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

# If we are on Mac, assume it is a local run
LOCAL_SYSTEM = (('mac' in platform.platform()) or ('Mac' in platform.platform()))

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# # NOTE - this is really useful, but will break parts of the
# # computation graph if NaNs are used for signalling purposes.
# # NaN debugging stuff.
# from jax.config import config
# config.update("jax_debug_nans", True)

DISABLE_JIT = False

PLOT = True  # NOTE - this will be overwritten.

# Uncomment this remove the functionality of the plotting code.
if not (LOCAL_SYSTEM and PLOT):
    # _plot_single_sweep = lambda *args, **kwargs: None
    # do_plot = lambda *args, **kwargs: None
    pass

# Import and configure WandB.
try:
    import wandb
    USE_WANDB = True
    USERNAME = 'andrewwarrington'
    PROJECT = 'fivo-aux-beta'
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

    # Set up the experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GDM', type=str)

    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--log-group', default='debug', type=str)               # {'debug', 'gdm-v1.0'}

    parser.add_argument('--use-sgr', default=1, type=int)                       # {0, 1}

    parser.add_argument('--free-parameters', default='dynamics_bias', type=str)              # CSV.  # 'dynamics_bias'
    parser.add_argument('--proposal-structure', default='DIRECT', type=str)       # {None/'BOOTSTRAP', 'DIRECT', 'RESQ', }
    parser.add_argument('--tilt-structure', default='DIRECT', type=str)         # {None/'NONE', 'DIRECT'}

    parser.add_argument('--num-particles', default=5, type=int)
    parser.add_argument('--datasets-per-batch', default=16, type=int)
    parser.add_argument('--opt-steps', default=100000, type=int)

    parser.add_argument('--p-lr', default=0.001, type=float)
    parser.add_argument('--q-lr', default=0.001, type=float)
    parser.add_argument('--r-lr', default=0.001, type=float)

    parser.add_argument('--dset-to-plot', default=2, type=int)
    parser.add_argument('--num-val-datasets', default=20, type=int)
    parser.add_argument('--validation-particles', default=100, type=int)
    parser.add_argument('--sweep-test-particles', default=20, type=int)

    parser.add_argument('--load-path', default=None, type=str)  # './params_lds_tmp.p'
    parser.add_argument('--save-path', default=None, type=str)  # './params_lds_tmp.p'

    parser.add_argument('--PLOT', default=1, type=int)

    config = parser.parse_args().__dict__

    # Write out to the global plot.
    global PLOT
    PLOT = config['PLOT']

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    if config['free_parameters'] is None or config['free_parameters'] == '':
        config['free_parameters'] = ()
    else:
        config['free_parameters'] = tuple(config['free_parameters'].split(','))

    # Do some type conversions.
    config['use_sgr'] = bool(config['use_sgr'])

    # Get everything.
    if USE_WANDB:
        # Set up WandB
        env = wandb.init(project=PROJECT, entity=USERNAME, group=config['log_group'], config=config)
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
    try:
        repo = git.Repo(search_parent_directories=True)
        env.config.git_commit = repo.head.object.hexsha
        env.config.git_branch = repo.active_branch
        env.config.git_is_dirty = repo.is_dirty()
    except:
        print('Failed to grab git info...')
        env.config.git_commit = 'NoneFound'
        env.config.git_branch = 'NoneFound'
        env.config.git_is_dirty = 'NoneFound'

    # Do some final bits.
    if len(env.config.free_parameters) == 0: print('\nWARNING: NO FREE MODEL PARAMETERS...\n')
    pprint(env)
    return env


def main():

    global _st

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Set up the experiment and log to WandB
        env = do_config()

        # Import the right functions.
        if env.config.model == 'GDM':
            from ssm.inference._test_fivo_gdm import gdm_do_print as do_print
            from ssm.inference._test_fivo_gdm import gdm_define_test as define_test
            from ssm.inference._test_fivo_gdm import gdm_do_plot as do_plot
            from ssm.inference._test_fivo_gdm import gdm_get_true_target_marginal as get_marginals

        elif env.config.model == 'LDS':
            from ssm.inference._test_fivo_lds import lds_do_print as do_print
            from ssm.inference._test_fivo_lds import lds_define_test as define_test
            from ssm.inference._test_fivo_lds import lds_do_plot as do_plot
            from ssm.inference._test_fivo_lds import lds_get_true_target_marginal as get_marginals

        else:
            raise NotImplementedError()

        # Define some holders that will be overwritten later.
        true_lml = 0.0
        em_log_marginal_likelihood = 0.0
        filt_fig = None
        sweep_fig_filter = None
        sweep_fig_smooth = None
        true_bpf_lml = 0.0
        true_bpf_kls = None
        true_bpf_upc = None

        # Set up the first key
        key = jr.PRNGKey(env.config.seed)

        # --------------------------------------------------------------------------------------------------------------

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey,
                               env.config.free_parameters,
                               env.config.proposal_structure,
                               env.config.tilt_structure)

        # Unpack that big mess of stuff.
        true_model, true_states, dataset = ret_vals[0]                  # Unpack true model.
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]    # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]        # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                # Unpack tilt.

        validation_datasets = dataset[:env.config.num_val_datasets]

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

        # --------------------------------------------------------------------------------------------------------------

        # Build up the optimizer.
        opt = fivo.define_optimizer(p_params=get_model_free_params(model),
                                    q_params=proposal_params,
                                    r_params=tilt_params,
                                    p_lr=env.config.p_lr,
                                    q_lr=env.config.q_lr,
                                    r_lr=env.config.r_lr,
                                    )

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
                               **{'use_stop_gradient_resampling': env.config.use_sgr})

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=(2, ))

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)


        # --------------------------------------------------------------------------------------------------------------

        # Do an e step.
        pred_em_posterior = jax.vmap(true_model.e_step)(validation_datasets)
        pred_em_lml = true_model.marginal_likelihood(validation_datasets, posterior=pred_em_posterior)
        pred_em_lml = - utils.lexp(pred_em_lml)

        # Test BPF in the true model..
        true_bpf_lml = []
        for _ in range(20):
            key, subkey = jr.split(key)
            _true_bpf_posterior = smc_jit(subkey, true_model, validation_datasets, num_particles=env.config.sweep_test_particles)
            _true_bpf_lml = - utils.lexp(_true_bpf_posterior.log_normalizer)
            true_bpf_lml.append(_true_bpf_lml)
        print('Variance: BPF-true: ', np.var(np.asarray(true_bpf_lml)))

        # Test the initial models.
        key, subkey = jr.split(key)
        true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound = \
            fivo.initial_validation(key,
                                    true_model,
                                    validation_datasets, true_states,
                                    opt,
                                    do_fivo_sweep_jitted,
                                    smc_jit,
                                    num_particles=env.config.validation_particles,
                                    dset_to_plot=env.config.dset_to_plot,
                                    init_model=model,
                                    do_print=do_print,
                                    do_plot=env.config.PLOT)

        # --------------------------------------------------------------------------------------------------------------

        # Wrap another call to fivo for validation.
        def _single_fivo_eval(_subkey, _params):
            _val_fivo_bound, _sweep_posteriors = do_fivo_sweep_jitted(_subkey,
                                                                      _params,
                                                                      _num_particles=env.config.sweep_test_particles,
                                                                      _datasets=validation_datasets)
            _val_fivo_lml = -utils.lexp(_sweep_posteriors.log_normalizer)
            return _val_fivo_lml, _val_fivo_bound

        single_fivo_eval_vmap = jax.vmap(_single_fivo_eval, in_axes=(0, None))

        # Define some storage.
        param_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        val_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        lml_hist = []
        param_figures = [None, None, None, None]  # Loss, Model, proposal, tilt.

        # Back up the true parameters.
        true_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, step.
        true_hist = fivo.log_params(true_hist,
                                    [get_model_free_params(true_model), None, None],
                                    true_lml,
                                    0.0,
                                    em_log_marginal_likelihood,
                                    0)

        # --------------------------------------------------------------------------------------------------------------

        # Main training loop.
        for _step in range(1, env.config.opt_steps + 1):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(env.config.datasets_per_batch, ), minval=0, maxval=len(dataset))
            batched_dataset = dataset.at[idx].get()

            # Do the sweep and compute the gradient.
            cur_params = dc(fivo.get_params_from_opt(opt))
            key, subkey = jr.split(key)
            (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                 fivo.get_params_from_opt(opt),
                                                                                 env.config.num_particles,
                                                                                 batched_dataset)
            pred_lml = - utils.lexp(smc_posteriors.log_normalizer)

            # Apply the gradient update.
            opt = fivo.apply_gradient(grad, opt, )

            # Log.
            if (_step % 10 == 0) or (len(param_hist[0]) == 0):
                lml_hist.append(dc(pred_lml))
                param_hist = fivo.log_params(param_hist, cur_params, pred_lml, pred_fivo_bound, 0.0, _step)

            # ----------------------------------------------------------------------------------------------------------

            # Do some validation and give some output.
            if (_step % 500 == 0) or (_step == 1):

                # Do a FIVO-AUX sweep.
                key, subkey = jr.split(key)
                pred_fivo_bound_to_print, pred_sweep = do_fivo_sweep_jitted(subkey,
                                                                            fivo.get_params_from_opt(opt),
                                                                            _num_particles=env.config.validation_particles,
                                                                            _datasets=validation_datasets)
                pred_lml_to_print = - utils.lexp(pred_sweep.log_normalizer)

                # Test the variance of the estimators.
                key, subkey = jr.split(key)
                val_fivo_lml, val_fivo_bound = single_fivo_eval_vmap(jr.split(key, 20), fivo.get_params_from_opt(opt))

                # Test the KLs.
                key, subkey = jr.split(key)
                true_bpf_kls, pred_smc_kls = fivo.compare_kls(get_marginals,
                                                              env,
                                                              opt,
                                                              validation_datasets,
                                                              true_model,
                                                              key,
                                                              do_fivo_sweep_jitted,
                                                              smc_jit,
                                                              true_bpf_kls=true_bpf_kls)

                # Compare the number of unique particles.
                key, subkey = jr.split(key)
                true_bpf_upc, pred_smc_upc = fivo.compare_unqiue_particle_counts(env,
                                                                                 opt,
                                                                                 validation_datasets,
                                                                                 true_model,
                                                                                 key,
                                                                                 do_fivo_sweep_jitted,
                                                                                 smc_jit,
                                                                                 true_bpf_upc=true_bpf_upc)

                # Do some plotting if we are plotting.
                if env.config.PLOT and (_step % 2000 == 0):

                    # Do some plotting.
                    sweep_fig_filter = _plot_single_sweep(
                        pred_sweep[env.config.dset_to_plot].filtering_particles,
                        true_states[env.config.dset_to_plot],
                        tag='{} Filtering.'.format(_step),
                        fig=sweep_fig_filter,
                        obs=dataset[env.config.dset_to_plot])
                    sweep_fig_smooth = _plot_single_sweep(
                        pred_sweep[env.config.dset_to_plot].weighted_smoothing_particles,
                        true_states[env.config.dset_to_plot],
                        tag='{} Smoothing.'.format(_step),
                        fig=sweep_fig_smooth,
                        obs=dataset[env.config.dset_to_plot])

                    param_figures = do_plot(param_hist,
                                            lml_hist,
                                            em_log_marginal_likelihood,
                                            true_lml,
                                            get_model_free_params(true_model),
                                            param_figures)

                    fivo.compare_sweeps(env, opt, validation_datasets, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key,
                                        do_fivo_sweep_jitted, smc_jit, tag=_step, nrep=5, true_states=true_states)

                # Log the validation step.
                val_hist = fivo.log_params(val_hist,
                                           cur_params,
                                           pred_lml_to_print,
                                           pred_fivo_bound_to_print,
                                           pred_em_lml,
                                           _step)

                # Do some printing.
                do_print(_step,
                         true_model,
                         opt,
                         true_lml,
                         pred_lml_to_print,
                         pred_fivo_bound_to_print,
                         em_log_marginal_likelihood)

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
                          'params_p_true': true_hist[0][-1],
                          'params_p_pred': param_hist[0][-1],
                          'params_q_pred': param_hist[1][-1],
                          'params_r_pred': param_hist[2][-1],
                          'pred_fivo_bound': pred_fivo_bound_to_print,
                          'pred_lml': pred_lml_to_print,
                          'small_lml_mean_em_true': em_log_marginal_likelihood,
                          'small_lml_mean_bpf_true': true_lml,
                          'small_lml_mean_fivo': np.mean(np.asarray(val_fivo_lml)),
                          'small_lml_variance_bpf_true': np.var(np.asarray(true_bpf_lml),),
                          'small_lml_variance_fivo': np.var(np.asarray(val_fivo_lml)),
                          'small_fivo_bound': np.mean(np.asarray(val_fivo_bound)),
                          'expected_kl_true': np.mean(true_bpf_kls),
                          'expected_kl_pred': np.mean(pred_smc_kls),
                          'expected_upc_bpf_true': np.mean(true_bpf_upc),
                          'expected_upc_fivo': np.mean(pred_smc_upc),
                          'upc_bpf_true': true_bpf_upc,
                          'upc_fivo': pred_smc_upc,
                          'true_lml': true_lml,
                          }
                utils.log_to_wandb(to_log, _epoch=_step, USE_WANDB=env.config.use_wandb)


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
