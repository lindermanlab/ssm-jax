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

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# # NOTE - this is really useful, but will break parts of the
# # computation graph if NaNs are used for signalling purposes.
# # NaN debugging stuff.
# from jax.config import config
# config.update("jax_debug_nans", True)

DISABLE_JIT = False

# If we are on Mac, assume it is a local run
local_system = (('mac' in platform.platform()) or ('Mac' in platform.platform()))

# from ssm.inference._test_fivo_lds import lds_do_print as do_print
# from ssm.inference._test_fivo_lds import lds_define_test as define_test
# from ssm.inference._test_fivo_lds import lds_do_plot as do_plot

from ssm.inference._test_fivo_gdm import gdm_do_print as do_print
from ssm.inference._test_fivo_gdm import gdm_define_test as define_test
from ssm.inference._test_fivo_gdm import gdm_do_plot as do_plot
from ssm.inference._test_fivo_gdm import gdm_get_true_target_marginal as get_marginals

# Uncomment this remove the functionality of the plotting code.
if (not local_system) and False:
    _plot_single_sweep = lambda *args, **kwargs: None
    do_plot = lambda *args, **kwargs: None

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
_verbose_clock_print = False
clock = lambda __st, __str: utils.clock(__st, __str, _verbose_clock_print)


def log_to_wandb(_dict=None, _ims=None, _epoch=None, _commit=True):
    """
    AW - Helper to push some info to WandB.

    The default behaviour is to push each time this fn is called, which steps on
    the counter inside WandB.  This can be surpressed by setting _commit=False to
    instead push on demand (by lastly calling log_to_wandb()) or push on the next
    time that this fn is called with _commit=True.  This means things can be
    logged from different subsections of the code and sitll be aligned.

    Also separates _dict, which should be

    :param _dict:
    :param _epoch:
    :param _commit:
    :param _ims:
    :return:
    """

    # Check if we are able to log.  If not, exit.
    if not USE_WANDB:
        return None

    _to_log = {}

    if _dict is not None:
        for _k in _dict:
            if (isinstance(_dict[_k], Iterable)) or (_dict[_k] is None):
                _to_log[_k] = _dict[_k]
            else:
                _to_log[_k] = float(_dict[_k])

    if _epoch is not None:
        _to_log['epoch'] = _epoch

    if _ims is not None:
        if type(_ims) is dict:
            _to_log = {**_to_log, **{_k: wandb.Image(_ims[_k]) for _k in _ims.keys()}}
        else:
            _to_log['image'] = wandb.Image(_ims)

    try:
        wandb.log(_to_log, commit=_commit, step=_epoch)
    except Exception as err:
        print('Error uploading to WandB: ', err)


def temp_validation_code(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _smc_jit,
                         _num_particles=10, _dset_to_plot=0, _init_model=None):

    # Do some sweeps.
    key, subkey = jr.split(key)
    smc_posterior = _smc_jit(subkey, true_model, dataset, num_particles=_num_particles)
    key, subkey = jr.split(key)
    initial_fivo_bound, sweep_posteriors = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                 _num_particles=_num_particles,
                                                                 _datasets=dataset)

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
        true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=_num_particles)
        true_bpf_lml = - utils.lexp(true_bpf_posterior.log_normalizer)
        val_bpf_lml.append(true_bpf_lml)

    for _ in range(20):
        key, subkey = jr.split(key)
        initial_fivo_bound, sweep_posteriors = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                     _num_particles=_num_particles,
                                                                     _datasets=dataset)
        initial_lml = -utils.lexp(sweep_posteriors.log_normalizer)
        val_fivo_lml.append(initial_lml)

    print('Variance: BPF:      ', np.var(np.asarray(val_bpf_lml)))
    print('Variance: FIVO-AUX: ', np.var(np.asarray(val_fivo_lml)))


def compute_marginal_kls(true_model, dataset, smoothing_particles):
    """

    E_q [ log Q / P ]

    Args:
        true_model:
        dataset:
        smoothing_particles:

    Returns:

    """

    eps = 1e-3

    # Get the analytic smoothing marginals.
    marginals = get_marginals(true_model, dataset)

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
    :param _smc_jit:
    :param _num_particles:
    :param _dset_to_plot:
    :param _init_model:
    :return:
    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0
    init_bpf_posterior = None

    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.e_step)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)

    # Test BPF in the true model..
    key, subkey = jr.split(key)
    true_bpf_posterior = _smc_jit(subkey, true_model, dataset, num_particles=_num_particles)
    true_lml = - utils.lexp(true_bpf_posterior.log_normalizer)

    if _init_model is not None:
        # Test BPF in the initial model..
        key, subkey = jr.split(key)
        init_bpf_posterior = _smc_jit(subkey, _init_model, dataset, num_particles=_num_particles)
        initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)
        print('Initial BPF LML: ', initial_bpf_lml)

    # Test SMC in the initial model.
    key, subkey = jr.split(key)
    initial_fivo_bound, init_smc_posterior = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                   _num_particles=_num_particles,
                                                                   _datasets=dataset)
    initial_lml = -utils.lexp(init_smc_posterior.log_normalizer)

    # # Dump any odd and ends of test code in here.
    # temp_validation_code(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _smc_jit,
    #                      _num_particles=10, _dset_to_plot=_dset_to_plot, _init_model=_init_model)

    # Do some plotting.
    sweep_em_mean = em_posterior.mean()[_dset_to_plot]
    sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[_dset_to_plot]
    sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    _plot_single_sweep(sweep_em_statistics, true_states[_dset_to_plot],
                       tag='EM smoothing', preprocessed=True, _obs=dataset[_dset_to_plot])

    _plot_single_sweep(true_bpf_posterior[_dset_to_plot].filtering_particles,
                       true_states[_dset_to_plot],
                       tag='True BPF Filtering.',
                       _obs=dataset[_dset_to_plot])
    _plot_single_sweep(true_bpf_posterior[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=jr.PRNGKey(0)),
                       true_states[_dset_to_plot],
                       tag='True BPF Smoothing.',
                       _obs=dataset[_dset_to_plot])

    if init_bpf_posterior is not None:
        _plot_single_sweep(init_bpf_posterior[_dset_to_plot].filtering_particles,
                           true_states[_dset_to_plot],
                           tag='Initial BPF Filtering.',
                           _obs=dataset[_dset_to_plot])
        _plot_single_sweep(init_bpf_posterior[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=jr.PRNGKey(0)),
                           true_states[_dset_to_plot],
                           tag='Initial BPF Smoothing.',
                           _obs=dataset[_dset_to_plot])

    filt_fig = _plot_single_sweep(init_smc_posterior[_dset_to_plot].filtering_particles,
                                  true_states[_dset_to_plot],
                                  tag='Initial SMC Filtering.',
                                  _obs=dataset[_dset_to_plot])
    sweep_fig = _plot_single_sweep(init_smc_posterior[_dset_to_plot].sample(sample_shape=(_num_particles,), seed=jr.PRNGKey(0)),
                                   true_states[_dset_to_plot],
                                   tag='Initial SMC Smoothing.',
                                   _obs=dataset[_dset_to_plot])

    # Do some print.
    do_print(0, true_model, opt, true_lml, initial_lml, initial_fivo_bound, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound


def _final_validation(env, opt, dataset, true_model, rebuild_model_fn, rebuild_prop_fn, rebuild_tilt_fn, key,
                      _do_fivo_sweep_jitted):

    # Do some final validation.
    # Rebuild the initial distribution.
    _prop = rebuild_prop_fn(fivo.get_params_from_opt(opt)[1])
    if _prop is not None:
        initial_distribution = lambda _dset, _model:  _prop(_dset, _model, np.zeros(dataset.shape[-1], ), 0, None, None)
    else:
        initial_distribution = None

    # BPF in true model.
    key, subkey = jr.split(key)
    final_val_posterior_bpf_true = smc(subkey,
                                       true_model,
                                       dataset,
                                       num_particles=env.config.sweep_test_particles)

    # SMC with tilt.
    key, subkey = jr.split(key)
    final_val_posterior_fivo_aux = smc(subkey,
                                       rebuild_model_fn(fivo.get_params_from_opt(opt)[0]),
                                       dataset,
                                       initialization_distribution=initial_distribution,
                                       proposal=rebuild_prop_fn(fivo.get_params_from_opt(opt)[1]),
                                       tilt=rebuild_tilt_fn(fivo.get_params_from_opt(opt)[2]),
                                       num_particles=env.config.sweep_test_particles)

    # CODE for plotting lineages.
    for _idx in range(10):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 8), tight_layout=True)

        for _p in final_val_posterior_bpf_true[_idx].weighted_smoothing_particles:
            ax[0].plot(_p, linewidth=0.1, c='b')
        ax[0].grid(True)
        ax[0].set_title('BPF in true model.')

        for _p in final_val_posterior_fivo_aux[_idx].weighted_smoothing_particles:
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

        plt.pause(0.01)
        plt.savefig('./tmp_sweep_{}_{}.pdf'.format(_tag, _idx))

    #
    #
    # Compare the KLs of the smoothing distributions.
    key, subkey = jr.split(key)
    true_bpf_posterior = smc(subkey, true_model, dataset, num_particles=env.config.sweep_test_particles)
    key, subkey = jr.split(key)
    _, pred_smc_posterior = _do_fivo_sweep_jitted(subkey,
                                                  fivo.get_params_from_opt(opt),
                                                  _num_particles=env.config.sweep_test_particles,
                                                  _datasets=dataset)

    true_bpf_kls = compute_marginal_kls(true_model, dataset, true_bpf_posterior.weighted_smoothing_particles)
    pred_smc_kls = compute_marginal_kls(true_model, dataset, pred_smc_posterior.weighted_smoothing_particles)
    # init_bpf_kls = compute_marginal_kls(true_model, dataset, init_bpf_posterior.weighted_smoothing_particles)

    plt.figure()
    plt.plot(np.median(np.asarray(true_bpf_kls), axis=1), label='True (BPF)')
    plt.plot(np.median(np.asarray(pred_smc_kls), axis=1), label='Pred (FIVO-AUX)')
    # plt.plot(np.median(np.asarray(init_bpf_kls), axis=1), label='bpf')
    plt.legend()
    plt.grid(True)
    plt.title('E_sweeps [ KL [ p_true[t] || q_pred[t] ] ]')
    plt.xlabel('Time, t')
    plt.ylabel('KL_t')
    plt.pause(0.001)
    plt.savefig('./kl_diff.pdf')
    #
    #


def do_config():
    """

    Returns:

    """

    default_seed = 10

    # Set up the experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=default_seed, type=int)
    parser.add_argument('--log-group', default='debug', type=str)               # {'debug', 'gdm-v1.0'}

    parser.add_argument('--proposal-structure', default='DIRECT', type=str)     # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    parser.add_argument('--tilt-structure', default='DIRECT', type=str)         # {'DIRECT', 'NONE'/None}
    parser.add_argument('--use-sgr', default=1, type=int)                       # {0, 1}

    parser.add_argument('--free-parameters', default='dynamics_bias', type=str)  # CSV.

    config = parser.parse_args().__dict__

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    config['free_parameters'] = tuple(config['free_parameters'].split(','))

    env = {  # Define some defaults.
        'dset_to_plot': 2,
        'num_val_datasets': 50,
        'validation_particles': 1000,
        'sweep_test_particles': 10,

        # Define the parameters to be used during optimization.
        'num_particles': 20,
        'opt_steps': 100000,
        'datasets_per_batch': 8,

        'load_path': None,  # './params_tmp.p'  # './params_tmp.p'  # {None, './params_tmp.p'}.
        'save_path': None,  # './params_tmp.p'  # {None, './params_tmp.p'}.

        # Add anything from the argparser  # TODO this should all be bumped into the argparser.
        **config}

    # do some type conversions.
    config['use_sgr'] = bool(config['use_sgr'])

    # Get everything.
    if log_to_wandb:
        # Set up WandB
        env = wandb.init(project=PROJECT, entity=USERNAME, group=env['log_group'], config=env)
    else:
        log_group = 'none'
        env = SimpleNamespace(**{'config': SimpleNamespace(**env),
                                 'log_group': log_group})

    # Set up some WandB stuff.
    env.config.log_to_wandb = bool(log_to_wandb)
    env.config.wandb_group = env.config.log_group
    env.config.wandb_project = PROJECT
    env.config.local_system = local_system

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

    pprint(env)
    return env


def main():

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Set up the experiment and log to WandB
        env = do_config()

        # Define some holders that will be overwritten later.
        true_lml = 0.0
        em_log_marginal_likelihood = 0.0
        filt_fig = None
        sweep_fig = None

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

        # Back up the true parameters.
        true_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, step.
        true_hist = fivo.log_params(true_hist,
                                    [get_model_free_params(true_model), None, None],
                                    true_lml,
                                    0.0,
                                    em_log_marginal_likelihood,
                                    0)

        # --------------------------------------------------------------------------------------------------------------

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
                               **{'use_stop_gradient_resampling': env.config.use_sgr})

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=(2, ))

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # --------------------------------------------------------------------------------------------------------------

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig, initial_lml, initial_fivo_bound = \
            initial_validation(key, true_model, validation_datasets, true_states, opt, do_fivo_sweep_jitted, smc_jit,
                               _num_particles=env.config.validation_particles, _dset_to_plot=env.config.dset_to_plot, _init_model=model)

        # Test BPF in the initial model..
        bpf = []
        for _ in range(20):
            key, subkey = jr.split(key)
            init_bpf_posterior = smc_jit(subkey, true_model, validation_datasets, num_particles=env.config.sweep_test_particles)
            initial_bpf_lml = - utils.lexp(init_bpf_posterior.log_normalizer)
            bpf.append(initial_bpf_lml)
        print('Variance: BPF: ', np.var(np.asarray(bpf)))

        # # TODO - TEMP
        # _final_validation(env,
        #                   opt,
        #                   validation_datasets,
        #                   true_model,
        #                   rebuild_model_fn,
        #                   rebuild_prop_fn,
        #                   rebuild_tilt_fn,
        #                   key)

        # --------------------------------------------------------------------------------------------------------------

        # Define some storage.
        param_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        val_hist = [[], [], [], [], [], [], []]  # Model, proposal, tilt, lml, fivo, em, step.
        lml_hist = []
        param_figures = [None, None, None, None]  # Loss, Model, proposal, tilt.

        # Main training loop.
        for _step in range(1, env.config.opt_steps + 1):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(env.config.datasets_per_batch, ), minval=0, maxval=len(dataset))
            batched_dataset = dataset.at[idx].get()

            # Do the sweep and compute the gradient.
            key, subkey = jr.split(key)
            cur_params = dc(fivo.get_params_from_opt(opt))
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
            if (_step % 2000 == 0) or (_step == 1):

                # Do an e step.
                pred_em_posterior = jax.vmap(true_model.e_step)(validation_datasets)
                pred_em_lml = true_model.marginal_likelihood(validation_datasets, posterior=pred_em_posterior)
                pred_em_lml = - utils.lexp(pred_em_lml)

                # Do a FIVO-AUX sweep.
                key, subkey = jr.split(key)
                pred_fivo_bound_to_print, pred_sweep = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                            _num_particles=env.config.validation_particles,
                                                                            _datasets=validation_datasets)
                pred_lml_to_print = - utils.lexp(pred_sweep.log_normalizer)

                # Test the variance of the estimators.
                val_fivo_lml = []
                for _ in range(20):
                    key, subkey = jr.split(key)
                    val_fivo_bound, sweep_posteriors = do_fivo_sweep_jitted(subkey,
                                                                            fivo.get_params_from_opt(opt),
                                                                            _num_particles=env.config.sweep_test_particles,
                                                                            _datasets=validation_datasets)
                    _val_fivo_lml = -utils.lexp(sweep_posteriors.log_normalizer)
                    val_fivo_lml.append(_val_fivo_lml)
                # print('Variance: FIVO-AUX: ', np.var(np.asarray(val_fivo_lml)))

                # Do some printing.
                do_print(_step,
                         true_model,
                         opt,
                         true_lml,
                         pred_lml_to_print,
                         pred_fivo_bound_to_print,
                         em_log_marginal_likelihood)

                # Do some plotting.
                sweep_fig = _plot_single_sweep(
                    pred_sweep[env.config.dset_to_plot].weighted_smoothing_particles,
                    true_states[env.config.dset_to_plot],
                    tag='{} Smoothing.'.format(_step),
                    fig=sweep_fig,
                    _obs=dataset[env.config.dset_to_plot])

                param_figures = do_plot(param_hist,
                                        lml_hist,
                                        em_log_marginal_likelihood,
                                        true_lml,
                                        get_model_free_params(true_model),
                                        param_figures)

                # Log the validation step.
                val_hist = fivo.log_params(val_hist,
                                           cur_params,
                                           pred_lml_to_print,
                                           pred_fivo_bound_to_print,
                                           pred_em_lml,
                                           _step)

                # Save out to a temporary location.
                if (env.config.save_path is not None) and (env.config.load_path is None):
                    with open(env.config.save_path, 'wb') as f:
                        params_to_dump = fivo.get_params_from_opt(opt)
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
                          'small_lml_variance_bpf_true': np.var(np.asarray(bpf),),
                          'small_lml_mean_em_true': em_log_marginal_likelihood,
                          'small_lml_mean_bpf_true': true_lml,
                          'small_lml_variance_fivo': np.var(np.asarray(val_fivo_lml)),
                          'small_lml_mean_fivo': np.mean(np.asarray(val_fivo_lml)),
                          'small_fivo_bound': np.mean(np.asarray(val_fivo_bound)),
                          }
                log_to_wandb(to_log, _epoch=_step)

        # Do some final validation.
        _final_validation(env,
                          opt,
                          validation_datasets,
                          true_model,
                          rebuild_model_fn,
                          rebuild_prop_fn,
                          rebuild_tilt_fn,
                          key,
                          do_fivo_sweep_jitted)


if __name__ == '__main__':
    main()
    print('Done')
