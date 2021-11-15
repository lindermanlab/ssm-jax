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
from ssm.utils import Verbosity, random_rotation
from ssm.inference.smc import smc, plot_single_sweep
from ssm.inference.em import em
import ssm.distributions as ssmd
from ssm.inference.snaxplicit import IndependentGaussianGenerator
import ssm.snax.snax as snax
from ssm.lds.models import GaussianLDS

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def get_params(_opt):
    return tuple(_o.target for _o in _opt)


def lexp(_lmls):
    _lml = spsp.logsumexp(_lmls) - len(_lmls)
    return _lml


def do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood=None):

    _str = 'Step: {: >5d},  True Neg-LML: {: >8.3f},  Pred Neg-LML: {: >8.3f}'.format(_step, true_lml, pred_lml)
    if em_log_marginal_likelihood is not None:
        _str += '  EM Neg-LML: {: >8.3f}'.format(em_log_marginal_likelihood)

    print(_str)
    print('True: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in true_model.dynamics_matrix.flatten()]))
    print('Pred: dynamics:  ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[0].target[0].flatten()]))
    print('True: log-var:   ', '  '.join(['{: >9.5f}'.format(_s) for _s in np.log(np.diagonal(true_model.dynamics_noise_covariance))]))
    print('Pred: q log-var: ', '  '.join(['{: >9.5f}'.format(_s) for _s in opt[1].target._asdict()['head_log_var_fn'].W.flatten()]))
    print()


def apply_gradient(full_loss_grad, optimizer, env=None, t=None):
    """

    :param full_loss_grad:
    :param optimizer:
    :return:
    """
    new_optimizer = [(_o.apply_gradient(_g) if _o is not None else None) for _o, _g in zip(optimizer, full_loss_grad)]
    return new_optimizer


def define_optimizer(p_params, q_params):
    """
    Build out the appropriate optimizer.

    :param p_params:
    :param q_params:
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

    opt = [p_opt, q_opt]
    return opt


def initial_validation(key, true_model, dataset, true_states, opt, _do_fivo_sweep_jitted):
    # Test against EM (which for the LDS is exact).
    em_posterior = jax.vmap(true_model.infer_posterior)(dataset)
    em_log_marginal_likelihood = true_model.marginal_likelihood(dataset, posterior=em_posterior)
    em_log_marginal_likelihood = - lexp(em_log_marginal_likelihood)
    sweep_em_mean = em_posterior.mean()[0]
    sweep_em_sds = np.sqrt(np.asarray([[np.diag(__k) for __k in _k] for _k in em_posterior.covariance()]))[0]
    sweep_em_statistics = (sweep_em_mean, sweep_em_mean - sweep_em_sds, sweep_em_mean + sweep_em_sds)
    plot_single_sweep(sweep_em_statistics, true_states[0], tag='EM smoothing', preprocessed=True)

    # Test SMC in the true model..
    key, subkey = jr.split(key)
    true_sweep, true_lml, _, _ = smc(subkey, true_model, dataset, num_particles=5000)
    true_lml = - lexp(true_lml)
    plot_single_sweep(true_sweep[0], true_states[0], tag='True Smoothing.')

    # Test SMC in the initial model.
    initial_params = dc(get_params(opt))
    key, subkey = jr.split(key)
    initial_lml, initial_sweep = _do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=5000)
    sweep_fig = plot_single_sweep(initial_sweep[0], true_states[0], tag='Initial Smoothing.')
    do_print(0, initial_lml, true_model, true_lml, opt, em_log_marginal_likelihood)

    return true_lml, em_log_marginal_likelihood, sweep_fig


def independent_gaussian_proposal(n_proposals, dummy_input, dummy_output,
                                  trunk_fn=None, head_mean_fn=None, head_log_var_fn=None):
    assert n_proposals == 1, 'Can only use a single proposal.'

    proposal = IndependentGaussianGenerator(dummy_input, dummy_output,
                                            trunk_fn=trunk_fn, head_mean_fn=head_mean_fn, head_log_var_fn=head_log_var_fn)

    def init(key):
        return proposal.init(key)

    def apply(params, inputs):
        vmapped = jax.vmap(snax.vectorize_pytree, in_axes=(None, 0))
        proposal_inputs = vmapped(jax.lax.dynamic_index_in_dim(inputs[0], index=0, axis=0, keepdims=False), inputs[2])
        return proposal(params, proposal_inputs)

    def get_output_dims(input_dims=None):
        return proposal.get_output_dims()

    return snax.Module(init, apply, get_output_dims)


def define_proposal(n_proposals, dummy_input, dummy_output):

    # Define a more conservative initialization.
    w_init = lambda *args: 0.1 * jax.nn.initializers.glorot_normal()(*args)
    output_dim = snax.vectorize_pytree(dummy_output).shape[0]

    # Use a static parameterization for the proposal covariance.
    head_log_var_fn = snax.Static(out_dim=output_dim, W_init=w_init)

    # Define a static proposal mean.
    head_mean_fn = snax.Static(out_dim=output_dim, W_init=w_init)

    return independent_gaussian_proposal(n_proposals, dummy_input, dummy_output,
                                         head_mean_fn=head_mean_fn, head_log_var_fn=head_log_var_fn)
    # return time_static_proposal(n_proposals, dummy_input, dummy_output)


def define_proposal_structure(PROPOSAL_STRUCTURE, proposal, _param_vals):

    # If there is no proposal, then there is no structure to define.
    if proposal is None:
        return None

    # We fork depending on the proposal type.
    # Proposal takes arguments of (dataset, model, particles, time, p_dist, ...).
    if PROPOSAL_STRUCTURE == 'DIRECT':

        def _proposal(*_input):
            z_dist = proposal.apply(_param_vals[1], _input)
            return z_dist

    elif PROPOSAL_STRUCTURE == 'RESQ':

        def _proposal(*_input):
            p_dist = _input[4]
            q_dist = proposal.apply(_param_vals[1], _input)
            z_dist = tfd.MultivariateNormalFullCovariance(loc=p_dist.mean() + q_dist.mean(),
                                                          covariance_matrix=q_dist.covariance())
            return z_dist
    else:
        raise NotImplementedError()

    return _proposal


def rebuild_model(_model, _param_vals, _p_params_accessors):
    for _v, _a in zip(_param_vals, _p_params_accessors):
        _model = _a(_model, _v)
    return _model


def do_fivo_sweep(_key, _model, _proposal, _param_vals, _p_params_accessors, _dataset, _num_particles, _prop_structure):
    # Reconstruct the model, inscribing the new parameter values.
    _model = rebuild_model(_model, _param_vals[0], _p_params_accessors)

    # Reconstruct the proposal function.
    _proposal = define_proposal_structure(_prop_structure, _proposal, _param_vals)

    # Do the sweep.
    _smooth, _lmls, _, _ = smc(_key, _model, _dataset, proposal=_proposal, num_particles=_num_particles)

    # Compute the log of the expected marginal.
    _lml = lexp(_lmls)
    return - _lml, _smooth


def main():

    key = jr.PRNGKey(2)

    # Set up true model and draw some data.
    latent_dim = 3
    emissions_dim = 5
    num_trials = 10
    num_timesteps = 100
    num_particles = 100
    opt_steps = 100000

    # Create a more reasonable emission scale.
    transition_scale_tril = 0.1 * np.eye(latent_dim)
    emission_scale_tril = 1.0 * np.eye(emissions_dim)

    # Create the true model.
    key, subkey = jr.split(key)
    true_dynamics_weights = random_rotation(subkey, latent_dim, theta=np.pi / 10)
    true_model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey,
                             dynamics_scale_tril=transition_scale_tril,
                             dynamics_weights=true_dynamics_weights,
                             emission_scale_tril=emission_scale_tril)

    # Sample some data.
    key, subkey = jr.split(key)
    true_states, dataset = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Now define a network to test.
    key, subkey = jax.random.split(key)
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey,
                        dynamics_scale_tril=transition_scale_tril,
                        emission_weights=true_model.emissions_matrix,
                        emission_scale_tril=emission_scale_tril)

    # Now define a proposal.
    PROPOSAL_STRUCTURE = 'RESQ'  # {'RESQ', 'DIRECT'}.
    proposal, proposal_params = None, None  # The default value for these is None.
    dummy_proposal_input = (np.ones((latent_dim, )), np.ones((emissions_dim, )))
    dummy_proposal_output = (np.ones((latent_dim, )), )
    proposal = define_proposal(n_proposals=1,
                               dummy_input=dummy_proposal_input,
                               dummy_output=dummy_proposal_output)
    key, subkey = jr.split(key)
    proposal_params = proposal.init(subkey)


    # TODO - KILL ME NOW.
    def accessor_0(_model, _param=None):
        if _param is None:
            return _model._dynamics._distribution.weights
        else:
            _model._dynamics._distribution = ssmd.GaussianLinearRegression(_param,
                                                                           _model._dynamics._distribution.bias,
                                                                           _model._dynamics._distribution.scale_tril)
            return _model

    p_params_accessors = (accessor_0, )
    p_params = tuple(_a(model) for _a in p_params_accessors)
    # TODO - END KILL ME NOW.

    # Build up the optimizer.
    opt = define_optimizer(p_params, proposal_params)

    # Close over constant parameters.
    do_fivo_sweep_closed = lambda _k, _p, _num_particles: \
        do_fivo_sweep(_k, model, proposal, _p, p_params_accessors, dataset, _num_particles, PROPOSAL_STRUCTURE)

    # Jit this badboy.
    do_fivo_sweep_jitted = jax.jit(do_fivo_sweep_closed, static_argnums=2)

    # Convert into value and grad.
    do_fivo_sweep_val_and_grad = jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

    # Test the initial models.
    true_lml, em_log_marginal_likelihood, sweep_fig = \
        initial_validation(key, true_model, dataset, true_states, opt, do_fivo_sweep_jitted)

    for _step in range(opt_steps):

        key, subkey = jr.split(key)
        (lml, smooth), grad = do_fivo_sweep_val_and_grad(subkey, get_params(opt), num_particles)
        opt = apply_gradient(grad, opt, )

        if _step % 1000 == 0:
            key, subkey = jr.split(key)
            pred_lml, pred_sweep = do_fivo_sweep_jitted(subkey, get_params(opt), _num_particles=5000)
            sweep_fig = plot_single_sweep(pred_sweep[0], true_states[0], tag='{} Smoothing.'.format(_step),
                                          fig=sweep_fig)
            do_print(_step, pred_lml, true_model, true_lml, opt, em_log_marginal_likelihood)
        p = 0

    print('Done')


if __name__ == '__main__':
    main()


