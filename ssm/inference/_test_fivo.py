"""
Wrapper for exploring FIVO..
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import random as jr
from copy import deepcopy as dc

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.inference.smc import smc, _plot_single_sweep
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

from ssm.inference._test_fivo_gdm import gdm_do_print as do_print
from ssm.inference._test_fivo_gdm import gdm_define_test as define_test
from ssm.inference._test_fivo_gdm import gdm_do_plot as do_plot


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _num_particles=5000, _dset_to_plot=0):
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
    true_lml, em_log_marginal_likelihood = 0.0, 0.0

    # # Test against EM (which for the LDS is exact).
    # em_posterior = jax.vmap(true_model.e_step)(dataset)
    # em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    # em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)
    # sweep_em_mean = em_posterior.mean()[_dset_to_plot]
    # sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[_dset_to_plot]
    # sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    # _plot_single_sweep(sweep_em_statistics, true_states[_dset_to_plot],
    #                    tag='EM smoothing', preprocessed=True, _obs=dataset[_dset_to_plot])

    # Test SMC in the true model..
    key, subkey = jr.split(key)
    smc_posterior = smc(subkey, true_model, dataset, num_particles=_num_particles)
    true_lml = - utils.lexp(smc_posterior.log_normalizer)
    _plot_single_sweep(smc_posterior.filtering_particles[_dset_to_plot], true_states[_dset_to_plot],
                       tag='True BPF Filtering.', _obs=dataset[_dset_to_plot])
    _plot_single_sweep(smc_posterior.particles[_dset_to_plot], true_states[_dset_to_plot],
                       tag='True BPF Smoothing.', _obs=dataset[_dset_to_plot])

    # Test SMC in the initial model.
    initial_params = dc(fivo.get_params_from_opt(opt))
    key, subkey = jr.split(key)
    initial_lml, sweep_posteriors = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                          _num_particles=_num_particles,
                                                          _num_datasets=len(dataset),
                                                          _datasets=dataset)
    filt_fig = _plot_single_sweep(sweep_posteriors.particles[_dset_to_plot], true_states[_dset_to_plot],
                                  tag='Initial Filtering.', _obs=dataset[_dset_to_plot])
    sweep_fig = _plot_single_sweep(sweep_posteriors.particles[_dset_to_plot], true_states[_dset_to_plot],
                                   tag='Initial Smoothing.', _obs=dataset[_dset_to_plot])

    # Do some print.
    do_print(0, initial_lml, true_model, true_lml, opt, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig


def log_params(_param_hist, _cur_params):

    if _cur_params[0] is not None:
        _param_hist[0].append(_cur_params[0]._asdict())
    else:
        _param_hist[0].append(None)

    if _cur_params[1] is not None:
        _p = _cur_params[1]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki
                if ('var' in _k) and ('bias' in _k):
                    _p_flat[_k + '_EXP'] = np.exp(_p[_ko][_ki])
                else:
                    _p_flat[_k] = _p[_ko][_ki]
        _param_hist[1].append(_p_flat)
    else:
        _param_hist[1].append(None)

    if _cur_params[2] is not None:
        _p = _cur_params[2]['params']._dict
        _p_flat = {}
        for _ko in _p.keys():
            for _ki in _p[_ko].keys():
                _k = _ko + '_' + _ki
                if ('var' in _k) and ('bias' in _k):
                    _p_flat[_k + '_EXP'] = np.exp(_p[_ko][_ki])
                else:
                    _p_flat[_k] = _p[_ko][_ki]
        _param_hist[2].append(_p_flat)
    else:
        _param_hist[2].append(None)

    return _param_hist


def main():

    # NOTE - FIVO should actually use multinomial resampling.  Therefore, test this.

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Define some defaults.
        key = jr.PRNGKey(1)
        true_lml = 0.0
        em_log_marginal_likelihood = 0.0
        filt_fig = None
        sweep_fig = None
        dset_to_plot = 2
        num_val_datasets = 100

        # Define the parameters to be used during optimization.
        USE_SGR = True
        num_particles = 25
        opt_steps = 100000
        datasets_per_batch = 4

        # Define the experiment.
        key, subkey = jr.split(key)
        ret_vals = define_test(subkey)

        # Unpack that big mess of stuff.
        true_model, true_states, dataset = ret_vals[0]                  # Unpack true model.
        model, get_model_free_params, rebuild_model_fn = ret_vals[1]    # Unpack test model.
        proposal, proposal_params, rebuild_prop_fn = ret_vals[2]        # Unpack proposal.
        tilt, tilt_params, rebuild_tilt_fn = ret_vals[3]                # Unpack tilt.

        # Build up the optimizer.
        opt = fivo.define_optimizer(p_params=get_model_free_params(model),
                                    q_params=proposal_params,
                                    r_params=tilt_params)

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _key, _params, _num_particles, _num_datasets, _datasets: \
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
            jax.jit(do_fivo_sweep_closed, static_argnums=(2, 3))

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig = \
            initial_validation(key, true_model, dataset[:num_val_datasets], true_states, opt, do_fivo_sweep_jitted,
                               _num_particles=5000, _dset_to_plot=dset_to_plot)

        # Define some storage.
        param_hist = [[], [], []]  # Model, proposal, tilt.
        loss_hist = []
        param_figures = [None, None, None]  # Model, proposal, tilt.

        # Main training loop.
        for _step in range(opt_steps):

            # Batch the data.
            key, subkey = jr.split(key)
            idx = jr.randint(key=subkey, shape=(datasets_per_batch, ), minval=0, maxval=len(dataset))
            batched_dataset = dataset.at[idx].get()

            # Do the sweep and compute the gradient.
            key, subkey = jr.split(key)
            cur_params = dc(fivo.get_params_from_opt(opt))
            (pred_lml, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                          fivo.get_params_from_opt(opt),
                                                                          num_particles,
                                                                          len(batched_dataset),
                                                                          batched_dataset)

            # Apply the gradient update.
            opt = fivo.apply_gradient(grad, opt, )

            # Log.
            loss_hist.append(dc(pred_lml))
            param_hist = log_params(param_hist, cur_params)

            # Do some validation and give some output.
            coldstart = 2
            if (_step % 2500 == 0) or (_step < coldstart):
                pred_lml_to_print = pred_lml

                if _step > coldstart:
                    key, subkey = jr.split(key)
                    pred_lml_to_print, pred_sweep = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                         _num_particles=5000,
                                                                         _num_datasets=len(dataset[:num_val_datasets]),
                                                                         _datasets=dataset[:num_val_datasets])
                    filt_fig = _plot_single_sweep(pred_sweep.filtering_particles[dset_to_plot], true_states[dset_to_plot],
                                                  tag='{} Filtering.'.format(_step), fig=filt_fig, _obs=dataset[dset_to_plot])
                    sweep_fig = _plot_single_sweep(pred_sweep.particles[dset_to_plot], true_states[dset_to_plot],
                                                   tag='{} Smoothing.'.format(_step), fig=sweep_fig, _obs=dataset[dset_to_plot])

                do_print(_step, pred_lml_to_print, true_model, true_lml, opt, em_log_marginal_likelihood)
                param_figures = do_plot(param_hist, loss_hist, em_log_marginal_likelihood, true_lml,
                                        get_model_free_params(true_model), param_figures)


if __name__ == '__main__':
    main()
    print('Done')

