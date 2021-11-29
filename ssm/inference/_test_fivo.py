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


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _num_particles=5000, _dset=0):
    """
    Do an test of the true model and the initialized model.
    :param key:
    :param true_model:
    :param dataset:
    :param true_states:
    :param opt:
    :param _do_fivo_sweep_jitted:
    :param _num_particles:
    :param _dset:
    :return:
    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0

    # # Test against EM (which for the LDS is exact).
    # em_posterior = jax.vmap(true_model.e_step)(dataset)
    # em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    # em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)
    # sweep_em_mean = em_posterior.mean()[_dset]
    # sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[_dset]
    # sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    # _plot_single_sweep(sweep_em_statistics, true_states[_dset],
    #                    tag='EM smoothing', preprocessed=True, _obs=dataset[_dset])

    # Test SMC in the true model..
    key, subkey = jr.split(key)
    smc_posterior = smc(subkey, true_model, dataset, num_particles=_num_particles)
    true_lml = - utils.lexp(smc_posterior.log_normalizer)
    _plot_single_sweep(smc_posterior.filtering_particles[_dset], true_states[_dset],
                       tag='True BPF Filtering.', _obs=dataset[_dset])
    _plot_single_sweep(smc_posterior.particles[_dset], true_states[_dset],
                       tag='True BPF Smoothing.', _obs=dataset[_dset])

    # Test SMC in the initial model.
    initial_params = dc(fivo.get_params_from_opt(opt))
    key, subkey = jr.split(key)
    initial_lml, sweep_posteriors = _do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                          _num_particles=_num_particles)
    filt_fig = _plot_single_sweep(sweep_posteriors.particles[_dset], true_states[_dset],
                                  tag='Initial Filtering.', _obs=dataset[_dset])
    sweep_fig = _plot_single_sweep(sweep_posteriors.particles[_dset], true_states[_dset],
                                   tag='Initial Smoothing.', _obs=dataset[_dset])

    # Do some print.
    do_print(0, initial_lml, true_model, true_lml, opt, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig


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
        dset = 1

        # Define the parameters to be used during optimization.
        USE_SGR = True
        num_particles = 25
        opt_steps = 100000

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
        do_fivo_sweep_closed = lambda _key, _params, _num_particles: \
            fivo.do_fivo_sweep(_params,
                               _key,
                               rebuild_model_fn,
                               rebuild_prop_fn,
                               rebuild_tilt_fn,
                               dataset,
                               _num_particles,
                               **{'use_stop_gradient_resampling': USE_SGR})

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=2)

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig, filt_fig = \
            initial_validation(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted,
                               _num_particles=5000, _dset=dset)

        # Main training loop.
        for _step in range(opt_steps):

            key, subkey = jr.split(key)
            (pred_lml, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                          fivo.get_params_from_opt(opt),
                                                                          num_particles)
            opt = fivo.apply_gradient(grad, opt, )

            coldstart = 2
            if (_step % 2500 == 0) or (_step < coldstart):

                if _step > coldstart:
                    pred_lml_to_print, pred_sweep = do_fivo_sweep_jitted(subkey, fivo.get_params_from_opt(opt),
                                                                         _num_particles=5000)
                    filt_fig = _plot_single_sweep(pred_sweep.filtering_particles[dset], true_states[dset],
                                                  tag='{} Filtering.'.format(_step), fig=filt_fig, _obs=dataset[dset])
                    sweep_fig = _plot_single_sweep(pred_sweep.particles[dset], true_states[dset],
                                                   tag='{} Smoothing.'.format(_step), fig=sweep_fig, _obs=dataset[dset])
                else:
                    pred_lml_to_print = pred_lml

                key, subkey = jr.split(key)
                do_print(_step, pred_lml_to_print, true_model, true_lml, opt, em_log_marginal_likelihood)


if __name__ == '__main__':
    main()
    print('Done')

