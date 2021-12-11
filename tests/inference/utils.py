import pytest
import tensorflow_probability.substrates.jax as tfp
from jax.interpreters import xla
from jax import vmap
import jax
import jax.scipy.special as spsp
import jax.random as jr
import jax.numpy as np
import pytest
from copy import deepcopy as dc

from ssm.inference.smc import smc
from ssm.inference import fivo
from ssm.hmm import BernoulliHMM
from ssm.lds import GaussianLDS
import ssm.utils as utils
import ssm.inference.proposals as proposals
import ssm.nn_util as nn_util
import flax.linen as nn


def single_smc_test(key, model, proposal, data, num_rounds, num_particles, ):
    """

    Args:
        key:
        model:
        proposal:
        data:
        num_rounds:
        num_particles:

    Returns:

    """

    # Test against EM (which for the LDS is exact.
    em_posterior = vmap(model.e_step)(data)
    em_log_marginal_likelihood = model.marginal_likelihood(data, posterior=em_posterior)

    # Close over a single SMC sweep.
    repeat_smc = lambda _k: smc(_k, model, data, proposal=proposal, num_particles=num_particles)
    vmapped = vmap(repeat_smc)

    # Run all the sweeps.
    key, subkey = jr.split(key)
    smc_posteriors = vmapped(jr.split(subkey, num=num_rounds))

    return em_log_marginal_likelihood, smc_posteriors


def construct_lds(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 3
    emissions_dim = 5
    model_kwargs = {'dynamics_scale_tril': 0.1 * np.eye(latent_dim),
                    'emission_scale_tril': 0.1 * np.eye(emissions_dim), }
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)
    return model


def construct_big_lds(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 5
    emissions_dim = 10
    model_kwargs = {'dynamics_scale_tril': 0.05 * np.eye(latent_dim),
                    'emission_scale_tril': 0.2 * np.eye(emissions_dim), }
    model = GaussianLDS(num_latent_dims=latent_dim, num_emission_dims=emissions_dim, seed=subkey, **model_kwargs)
    return model


def construct_bhmm(subkey):
    """

    Args:
        subkey:

    Returns:

    """
    latent_dim = 2
    emissions_dim = 3
    key, subkey = jr.split(subkey)
    model = BernoulliHMM(num_states=latent_dim, num_emission_dims=emissions_dim, seed=subkey)
    return model


def construct_default_proposal(subkey, model, data):
    """
    We don't have proposals implemented for discrete models yet.
    Args:
        subkey:
        model:
        data:

    Returns:

    """
    return None, None, (lambda *args: None)


def construct_lds_proposal(subkey, model, dataset):
    """
    Define the proposal that we will use in FIVO learning.

    Args:
        subkey:
        model:
        dataset:

    Returns:

    """
    proposal_structure = 'RESQ'  # {None/'BOOTSTRAP', 'RESQ', 'DIRECT', }
    proposal_type = 'SHARED'  # {'SHARED', 'INDPENDENT'}

    if (proposal_structure is None) or (proposal_structure == 'BOOTSTRAP'):
        _empty_rebuild = lambda *args: None
        return None, None, _empty_rebuild

    # Define the proposal that we will use.
    # Stock proposal input form is (dataset, model, particles, t, p_dist, q_state).
    dummy_particles = model.initial_distribution().sample(seed=jr.PRNGKey(0), sample_shape=(2,), )
    dummy_p_dist = model.dynamics_distribution(dummy_particles)
    stock_proposal_input_without_q_state = (dataset[0], model, dummy_particles[0], 0, dummy_p_dist)
    dummy_proposal_output = nn_util.vectorize_pytree(np.ones((model.latent_dim,)), )
    output_dim = nn_util.vectorize_pytree(dummy_proposal_output).shape[0]

    w_init_mean = lambda *args: (0.1 * jax.nn.initializers.normal()(*args))
    head_mean_fn = nn.Dense(output_dim, kernel_init=w_init_mean)

    w_init_mean = lambda *args: ((0.1 * jax.nn.initializers.normal()(*args)) - 3)
    head_log_var_fn = nn_util.Static(output_dim, kernel_init=w_init_mean)

    # Define the number of proposals to define depending on the proposal type.
    if proposal_type == 'SHARED':
        n_proposals = 1
    else:
        n_proposals = len(dataset)

    # Define the proposal itself.
    proposal = proposals.IndependentGaussianProposal(n_proposals=n_proposals,
                                                     stock_proposal_input_without_q_state=stock_proposal_input_without_q_state,
                                                     dummy_output=dummy_proposal_output,
                                                     head_mean_fn=head_mean_fn,
                                                     head_log_var_fn=head_log_var_fn, )
    proposal_params = proposal.init(subkey)

    # Return a function that we can call with just the parameters as an argument to return a new closed proposal.
    rebuild_prop_fn = proposals.rebuild_proposal(proposal, proposal_structure)
    return proposal, proposal_params, rebuild_prop_fn



def lds_define_test_model(subkey, true_model, ):
    """
    Define the model in which we will perform FIVO parameter learning.

    Args:
        subkey:
        true_model:

    Returns:

    """

    # Define the parameter names that we are going to learn.
    # This has to be a tuple of strings that index which args we will pull out.
    free_parameters = ('dynamics_weights', )

    # Close over the free parameters we have elected to learn.
    get_free_model_params_fn = lambda _model: fivo.get_model_params_fn(_model, free_parameters)

    if len(free_parameters) > 0:

        # Get the default parameters from the true model.
        true_params = fivo.get_model_params_fn(true_model)

        # Generate a model to use.  NOTE - this will generate a new model, and we will
        # overwrite any of the free parameters of interest into the true model.
        tmp_model = true_model.__class__(num_latent_dims=true_model.latent_dim,
                                         num_emission_dims=true_model.emissions_shape[0],
                                         seed=subkey)

        # Dig out the free parameters.
        init_free_params = get_free_model_params_fn(tmp_model)

        # Overwrite all the params with the new values.
        default_params = utils.mutate_named_tuple_by_key(true_params, init_free_params)

        # Build out a new model using these values.
        default_model = fivo.rebuild_model_fn(default_params, tmp_model)

    else:

        # If there are no free parameters then just use the true model.
        default_model = dc(true_model)

    # Close over rebuilding the model.
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, default_model)

    return default_model, get_free_model_params_fn, rebuild_model_fn


def run_fivo(key,
             _tag,
             _model_constructor,
             _proposal_constructor,
             _n_opt_steps=1,
             _verbose=False):
    """
    Verify that fivo still runs.
    Args:
        _tag:
        _model_constructor:
        _proposal_constructor:
        _n_opt_steps:
        _verbose:

    Returns:

    """

    # Define the model.
    key, subkey = jr.split(key)
    true_model = _model_constructor(subkey)

    # Sample the data.
    num_trials = 200
    num_timesteps = 100
    num_particles = 10
    datasets_per_batch = 4
    num_val_particles = 1000
    key, subkey = jr.split(key)
    true_states, data = true_model.sample(key=subkey, num_steps=num_timesteps, num_samples=num_trials)

    # Test against EM (which for the LDS is exact.
    em_posterior = vmap(true_model.e_step)(data)
    em_nlml = - utils.lexp(true_model.marginal_likelihood(data, posterior=em_posterior))

    # Define the model we will test in.
    key, subkey = jr.split(key)
    model, get_model_params, rebuild_model_fn = lds_define_test_model(subkey, true_model)
    rebuild_model_fn = lambda _params: fivo.rebuild_model_fn(_params, model)

    # Define the proposal to use.
    key, subkey = jr.split(key)
    proposal, proposal_params, rebuild_prop_fn = _proposal_constructor(subkey, model, data)

    # Build up the FIVO scripts.
    # Close over constant parameters.
    do_fivo_sweep_closed = lambda _key, _params, _num_particles, _datasets: \
        fivo.do_fivo_sweep(_params,
                           _key,
                           rebuild_model_fn,
                           rebuild_prop_fn,
                           _datasets,
                           _num_particles,
                           **{})

    # Build up the optimizer.
    opt = fivo.define_optimizer(p_params=get_model_params(model),
                                q_params=proposal_params,
                                p_lr=0.01,
                                q_lr=0.01)

    # Jit this badboy.
    do_fivo_sweep_jitted = \
        jax.jit(do_fivo_sweep_closed, static_argnums=(2, ))

    # Convert into value and grad.
    do_fivo_sweep_val_and_grad = \
        jax.value_and_grad(do_fivo_sweep_jitted, argnums=1, has_aux=True)

    # Do the sweep and compute the gradient.
    for _n in range(_n_opt_steps):
        # Batch the data.
        key, subkey = jr.split(key)
        idx = jr.randint(key=subkey, shape=(datasets_per_batch,), minval=0, maxval=len(data))
        batched_dataset = data.at[idx].get()

        key, subkey = jr.split(key)
        _, grad = do_fivo_sweep_val_and_grad(subkey, fivo.get_params_from_opt(opt), num_particles, batched_dataset)

        # Apply the gradient update.
        opt = fivo.apply_gradient(grad, opt, )

        if _n % 1000 == 0:

            # Do a final sweep.
            (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                                 fivo.get_params_from_opt(opt),
                                                                                 num_val_particles,
                                                                                 data)
            pred_nlml = - utils.lexp(smc_posteriors.log_normalizer)
            if _verbose:
                print(_n, "NLML: True: {: >6.2f}".format(em_nlml), " Pred: {: >6.2f}".format(pred_nlml),
                      "FIVO: {: >6.2f}".format(pred_fivo_bound))

    # Do a final sweep.
    (pred_fivo_bound, smc_posteriors), grad = do_fivo_sweep_val_and_grad(subkey,
                                                                         fivo.get_params_from_opt(opt),
                                                                         num_val_particles,
                                                                         data)
    pred_nlml = - utils.lexp(smc_posteriors.log_normalizer)

    return get_model_params(true_model), em_nlml, pred_nlml, fivo.get_params_from_opt(opt)
