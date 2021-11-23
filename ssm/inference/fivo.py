"""
FIVO implementation for join state-space inference and parameter learning in SSMs.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import NamedTuple, Any
from flax import optim
from copy import deepcopy as dc

# Import some ssm stuff.
from ssm.utils import Verbosity, random_rotation, possibly_disable_jit
from ssm.inference.smc import smc, _plot_single_sweep
import ssm.distributions as ssmd
from ssm.lds.models import GaussianLDS
import ssm.nn_util as nn_util
import ssm.utils as utils
import ssm.inference.proposals as proposals

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG

# NOTE - this is really useful, but will break parts of the
# computation graph if NaNs are used for signalling purposes.
# NaN debugging stuff.
from jax.config import config
config.update("jax_debug_nans", True)

DISABLE_JIT = False


def get_params(_opt):
    """
    Pull the parameters (stored in the Flax optimizer target) out of the optimizer tuple.
    :param _opt: Tuple of Flax optimizer objects.
    :return: Tuple of parameters.
    """
    return tuple((_o.target if _o is not None else None) for _o in _opt)


def do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood=None):
    """
    Do menial print stuff.
    :param _step:
    :param pred_lml:
    :param true_model:
    :param true_lml:
    :param opt:
    :param em_log_marginal_likelihood:
    :return:
    """

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f}'.format(_step, true_lml, pred_lml)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    print('True: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in true_model.dynamics_matrix.flatten()]))
    print('Pred: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[0].target[0].flatten()]))

    # if opt[1] is not None:
    #     print('True: log-var:   ', '  '.join(['{: >9.5f}'.format(_s) for _s in np.log(np.diagonal(true_model.dynamics_noise_covariance))]))
    #     print('Pred: q log-var: ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[1].target._asdict()['head_log_var_fn'].W.flatten()]))
    print()


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted, _num_particles=5000, dset=0):
    """
    Do an test of the true model and the initialized model.
    :param key:
    :param true_model:
    :param dataset:
    :param true_states:
    :param opt:
    :param _do_fivo_sweep_jitted:
    :param _num_particles:
    :param dset:
    :return:
    """
    true_lml, em_log_marginal_likelihood = 0.0, 0.0

    dset = 0

    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.infer_posterior)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - utils.lexp(em_log_marginal_likelihood)
    sweep_em_mean = em_posterior.mean()[dset]
    sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[dset]
    sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    _plot_single_sweep(sweep_em_statistics, true_states[0], tag='EM smoothing', preprocessed=True)

    # Test SMC in the true model..
    key, subkey = jr.split(key)
    smc_posterior = smc(subkey, true_model, dataset, num_particles=_num_particles)
    true_sweep, true_lml = smc_posterior.particles, smc_posterior.log_normalizer
    true_lml = - utils.lexp(true_lml)
    _plot_single_sweep(true_sweep[dset], true_states[dset], tag='True Smoothing.')

    # Test SMC in the initial model.
    initial_params = dc(get_params(opt))
    key, subkey = jr.split(key)
    initial_lml, sweep_posteriors = _do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=_num_particles)
    sweep_fig = _plot_single_sweep(sweep_posteriors.particles[dset], true_states[dset], tag='Initial Smoothing.')
    do_print(0, initial_lml, true_model, true_lml, opt, em_log_marginal_likelihood)
    return true_lml, em_log_marginal_likelihood, sweep_fig


def apply_gradient(full_loss_grad, optimizer):
    """
    Apply the optimization update to the parameters using the gradient.

    full_loss_grad and optimizer must be tuples of the same pytrees.  I.e., grad[4] will be passed into opt[4].

    The optimizer can be None, in which case there is no gradient update applied.

    :param full_loss_grad:      Tuple of gradients, each formatted as an arbitrary pytree.
    :param optimizer:           Tuple of optimizers, one for each entry in full_loss_grad
    :return:                    Updated tuple of optimizers.
    """
    new_optimizer = [(_o.apply_gradient(_g) if _o is not None else None) for _o, _g in zip(optimizer, full_loss_grad)]
    return new_optimizer


def define_optimizer(p_params=None, q_params=None, r_params=None):
    """
    Build out the appropriate optimizer.

    If an inputs is None, then no optimizer is defined and a None flag is used instead.

    :param p_params:    Pytree of the parameters of the SSM.
    :param q_params:    PyTree of the parameters of the proposal.
    :param r_params:    PyTree of the parameters of the tilt.
    :return:
    """

    if p_params is not None:
        p_opt_def = optim.Adam(learning_rate=0.001)
        p_opt = p_opt_def.create(p_params)
    else:
        p_opt = None

    if q_params is not None:
        q_opt_def = optim.Adam(learning_rate=0.01)
        q_opt = q_opt_def.create(q_params)
    else:
        q_opt = None

    if r_params is not None:
        r_opt_def = optim.Adam(learning_rate=0.01)
        r_opt = r_opt_def.create(r_params)
    else:
        r_opt = None

    opt = [p_opt, q_opt, r_opt]
    return opt


def do_fivo_sweep(_param_vals, _key, _rebuild_model, _rebuild_proposal, _dataset, _num_particles):

    # Reconstruct the model, inscribing the new parameter values.
    _model = _rebuild_model(_param_vals[0])

    # Reconstruct the proposal.
    _proposal = _rebuild_proposal(_param_vals[1])

    # Do the sweep.
    _smc_posteriors = smc(_key, _model, _dataset, proposal=_proposal, num_particles=_num_particles)

    # Compute the log of the expected marginal.
    _lml = utils.lexp(_smc_posteriors.log_normalizer)

    return - _lml, _smc_posteriors


def define_rebuild_model(_model, _p_params_accessors):
    """
    This function can take anything as arguments, but MUST return a function that takes EXACTLY the parameters of the
    model and in turn returns the model updated with the supplied parameters..

    # TODO - this paradigm may need updating.

    :param _model:
    :param _p_params_accessors:
    :return:
    """

    def rebuild_model(_param_vals):
        _rebuilt_model = dc(_model)
        for _v, _a in zip(_param_vals, _p_params_accessors):
            _rebuilt_model = _a(_rebuilt_model, _v)
        return _rebuilt_model

    return rebuild_model


def define_true_model_and_data(key):
    """
    
    :param key: 
    :return: 
    """
    latent_dim = 3
    emissions_dim = 5
    num_trials = 10
    num_timesteps = 100

    # Create a more reasonable emission scale.
    transition_scale_tril = 0.1 * np.eye(latent_dim)
    emission_scale_tril = 0.5 * np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)
    true_dynamics_weights = random_rotation(subkey, latent_dim, theta=np.pi / 10)
    true_model = GaussianLDS(num_latent_dims=latent_dim,
                             num_emission_dims=emissions_dim,
                             seed=subkey,
                             dynamics_scale_tril=transition_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    return true_model, true_states, dataset


def define_test_model(subkey, true_model, ):
    """
    TODO - This paradigm needs a lot of work.
    :param subkey:
    :param true_model:
    :return:
    """
    model = GaussianLDS(num_latent_dims=true_model.latent_dim,
                        num_emission_dims=true_model.emissions_dim,
                        seed=subkey,
                        dynamics_scale_tril=true_model._dynamics.scale_tril,
                        emission_weights=true_model.emissions_matrix,
                        emission_scale_tril=true_model._emissions.scale_tril)

    # TBEGIN PAIN AND MISERY AND DEATH AND DESTRUCTION AND SADNESS.
    def accessor_0(_model, _param=None):
        if _param is None:
            return _model._dynamics._distribution.weights
        else:
            _model._dynamics._distribution = ssmd.GaussianLinearRegression(_param,
                                                                           _model._dynamics._distribution.bias,
                                                                           _model._dynamics._distribution.scale_tril)
            return _model

    p_params_accessors = (accessor_0,)
    p_params = tuple(_a(model) for _a in p_params_accessors)
    return model, p_params, p_params_accessors


def define_proposal(subkey, model, dataset):
    """

    :param subkey:
    :param model:
    :param dataset:
    :return:
    """
    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist).
    proposal_structure = 'RESQ'  # {'RESQ', 'DIRECT'}.
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )
    proposal = proposals.IndependentGaussianProposal(n_proposals=1,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output)
    proposal_params = proposal.init(subkey)
    return proposal_structure, proposal_params


def main():

    # Give the option of disabling JIT to allow for better inspection and debugging.
    with possibly_disable_jit(DISABLE_JIT):

        # Define some defaults.
        key = jr.PRNGKey(2)
        proposal, proposal_params, proposal_structure = None, None, None
        tilt, tilt_params, tilt_structure = None, None, None

        # Define the true model.
        key, subkey = jr.split(key)
        true_model, true_states, dataset = define_true_model_and_data(subkey)

        # Now define a model to test.
        key, subkey = jax.random.split(key)
        model, p_params, p_params_accessors = define_test_model(subkey, true_model)

        # Define the proposal.
        key, subkey = jr.split(key)
        proposal_structure, proposal_params = define_proposal(subkey, model, dataset)

        # Define the functions for constructing objects.
        rebuild_model_fn = define_rebuild_model(model, p_params_accessors)
        rebuild_prop_fn = proposals.wrap_proposal_structure(proposal, proposal_structure)

        # Build up the optimizer.
        opt = define_optimizer(p_params, proposal_params)

        # Close over constant parameters.
        do_fivo_sweep_closed = lambda _k, _p, _num_particles: \
            do_fivo_sweep(_p, _k, rebuild_model_fn, rebuild_prop_fn, dataset, _num_particles)

        # Jit this badboy.
        do_fivo_sweep_jitted = \
            jax.jit(do_fivo_sweep_closed, static_argnums=2)

        # Convert into value and grad.
        do_fivo_sweep_val_and_grad = \
            jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

        # Test the initial models.
        true_lml, em_log_marginal_likelihood, sweep_fig = \
            initial_validation(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted, _num_particles=5000)

        # Define the parameters to be used during optimization.
        num_particles = 100
        opt_steps = 100000

        # Main training loop.
        for _step in range(opt_steps):

            key, subkey = jr.split(key)
            (pred_lml, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey, get_params(opt), num_particles)
            opt = apply_gradient(grad, opt, )

            coldstart = 10
            if (_step % 1000 == 0) or (_step < coldstart):
                key, subkey = jr.split(key)
                do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood)

                if _step > coldstart:
                    pred_lml, pred_sweep = do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=5000)
                    sweep_fig = _plot_single_sweep(pred_sweep.particles[0], true_states[0],
                                                   tag='{} Smoothing.'.format(_step), fig=sweep_fig)


if __name__ == '__main__':
    main()
    print('Done')

