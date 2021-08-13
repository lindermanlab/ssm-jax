"""
TODO: organize HMM-specific em functions elsewhere
"""

import warnings
from collections import namedtuple
from enum import IntEnum
from functools import partial, wraps
from textwrap import dedent

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
import matplotlib.pyplot as plt
from jax import grad, jit, lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node, register_pytree_node_class
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

from .core.hmm_message_passing import hmm_expected_states
from .priors import EXPFAM_DISTRIBUTIONS
from .utils import Verbosity, ssm_pbar

HMMPosterior = namedtuple(
    "HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"]
)


def sum_tuples(a, b):
    assert a or b
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(ai + bi for ai, bi in zip(a, b))


def _e_step(hmm, data):
    log_initial_state_distn = hmm.initial_state_dist().logits_parameter()
    log_transition_matrix = np.log(hmm.transition_matrix)
    log_likelihoods = vmap(
        lambda k: vmap(lambda x: hmm.emissions_dist(k).log_prob(x))(data)
    )(np.arange(hmm.num_states)).T

    marginal_likelihood, (Ez0, Ezzp1, Ez) = hmm_expected_states(
        log_initial_state_distn, log_transition_matrix, log_likelihoods
    )

    return HMMPosterior(marginal_likelihood, Ez, Ezzp1)


def _exact_m_step_initial_state_dist(hmm, data, posterior):
    expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
    prior_stats, prior_counts = expfam.prior_pseudo_obs_and_counts(
        hmm._initial_state_prior
    )
    lkhd_stats, lkhd_counts = (posterior.expected_states[0],), 0
    param_posterior = expfam.posterior_from_stats(
        sum_tuples(prior_stats, lkhd_stats), prior_counts + lkhd_counts
    )
    hmm._initial_state_dist = expfam.from_params(param_posterior.mode())


def _exact_m_step_dynamics_dist(hmm, data, posterior):
    expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
    prior_stats, prior_counts = expfam.prior_pseudo_obs_and_counts(hmm._dynamics_prior)
    lkhd_stats, lkhd_counts = (posterior.expected_transitions,), 0
    param_posterior = expfam.posterior_from_stats(
        sum_tuples(prior_stats, lkhd_stats), prior_counts + lkhd_counts
    )
    hmm._dynamics_dist = expfam.from_params(param_posterior.mode())


def _exact_m_step_emissions_dist(hmm, data, posterior):
    # Use exponential family stuff for the emissions
    expfam = EXPFAM_DISTRIBUTIONS[hmm._emissions_dist.name]

    # TODO: incorporate prior

    # Compute sufficient statistics of the data
    stats = expfam.suff_stats(data)
    weighted_stats = tuple(
        np.einsum("tk,t...->k...", posterior.expected_states, s) for s in stats
    )
    counts = np.sum(posterior.expected_states, axis=0)

    param_posterior = expfam.posterior_from_stats(weighted_stats, counts)
    hmm._emissions_dist = expfam.from_params(param_posterior.mode())


def _exact_m_step(hmm, data, posterior):
    _exact_m_step_initial_state_dist(hmm, data, posterior)
    _exact_m_step_dynamics_dist(hmm, data, posterior)
    _exact_m_step_emissions_dist(hmm, data, posterior)


def hmm_marginal_likelihood(hmm, data):
    posterior = _e_step(hmm, data)
    return posterior.marginal_likelihood


def _generic_m_step(hmm, data, posterior, num_iters=1, lr=1e-3):
    _objective = lambda hmm, data: -hmm_marginal_likelihood(hmm, data) / data.size
    g = grad(_objective, argnums=0)

    # @jit
    def _step(hmm, data):
        gradient = g(hmm, data)

        # treat all children as parameters to update via sgd
        hmm_children, hmm_aux_data = hmm.tree_flatten()
        gradient_children, _ = gradient.tree_flatten()
        new_hmm_children = []
        for (hmm_child, gradient_child) in zip(hmm_children, gradient_children):
            new_hmm_children.append(hmm_child - lr * gradient_child)
        hmm.tree_unflatten_inplace(hmm_aux_data, new_hmm_children)

        # return hmm
        # return hmm.tree_unflatten(hmm_aux_data, tuple(new_hmm_children))

        # # using unflatten directly breaks pass by reference :(
        # # [if uncommenting, make sure you plumb the new_hmm thru via returns]
        # new_hmm = hmm.tree_unflatten(hmm_aux_data, tuple(new_hmm_children))
        # return new_hmm

        # hmm._initial_state_dist = hmm._initial_state_dist.__class__(
        #     logits=hmm._initial_state_dist.logits_parameter()
        #     - lr * gradient._initial_state_dist.logits_parameter()
        # )
        # hmm._dynamics_dist = hmm._dynamics_dist.__class__(
        #     hmm._dynamics_dist.logits_parameter()
        #     - lr * gradient._dynamics_dist.logits_parameter()
        # )
        # new_loc = hmm._emissions_dist.loc - lr * gradient._emissions_dist.loc
        # new_scale_tril = (
        #     hmm._emissions_dist.scale_tril - lr * gradient._emissions_dist.scale_tril
        # )
        # hmm._emissions_dist = hmm._emissions_dist.__class__(
        #     loc=new_loc, scale_tril=new_scale_tril
        # )

    for i in range(num_iters):
        _step(hmm, data)


def _generic_m_step_elbo(hmm, data, posterior, num_iters=1):

    # Update the initial distribution and the transition distribution exactly
    _exact_m_step_initial_state_dist(hmm, data, posterior)
    _exact_m_step_dynamics_dist(hmm, data, posterior)

    def _expected_log_joint(hmm):
        return np.sum(
            posterior.expected_states * hmm._emissions_dist.log_prob(data[:, None, :])
        )

    def _objective(hmm):
        return -_expected_log_joint(hmm) / data.size

    g = grad(_objective)

    # SGD
    lr = 5e-3
    for i in range(num_iters):
        gradient = g(hmm)

        # update parameter values (hacky for now...)
        # idea: filter children for emissions_dist and update only those
        # then somehow reconstruct in a class-type-agnostic manner
        if isinstance(hmm._emissions_dist, tfp.distributions.MultivariateNormalTriL):
            new_loc = hmm._emissions_dist.loc - lr * gradient._emissions_dist.loc
            new_scale = (
                hmm._emissions_dist.scale_tril
                - lr * gradient._emissions_dist.scale_tril
            )
            hmm._emissions_dist = tfp.distributions.MultivariateNormalTriL(
                loc=new_loc, scale_tril=new_scale
            )
        elif isinstance(hmm._emissions_dist, tfp.distributions.Independent):
            new_log_rate = (
                hmm._emissions_dist.distribution.log_rate_parameter()
                - lr * gradient._emissions_dist.distribution.log_rate_parameter()
            )
            hmm._emissions_dist = tfp.distributions.Independent(
                tfp.distributions.Poisson(log_rate=new_log_rate),
                reinterpreted_batch_ndims=1,
            )


def em(
    hmm,
    data,
    num_iters=100,
    tol=1e-4,
    verbosity=Verbosity.DEBUG,
    m_step_type="exact",
    num_inner=1,
    patience=5,
):
    @jit
    def step(hmm):
        posterior = _e_step(hmm, data)
        if m_step_type == "exact":
            _exact_m_step(hmm, data, posterior)
        elif m_step_type == "sgd_marginal_likelihood":
            _generic_m_step(hmm, data, posterior, num_iters=num_inner)
        elif m_step_type == "sgd_expected_log_prob":
            _generic_m_step_elbo(hmm, data, posterior, num_iters=num_inner)
        else:
            raise ValueError("unrecognized")
        return hmm, posterior

    # Run the EM algorithm to convergence
    log_probs = [np.nan]
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, log_probs[-1])
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience
    for itr in pbar:
        hmm, posterior = step(hmm)
        lp = posterior.marginal_likelihood
        log_probs.append(lp)
        if verbosity:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        # TODO: make this cleaner with patience
        if abs(log_probs[-1] - log_probs[-2]) < tol and itr > 1:
            if patience == 0:
                if verbosity:
                    pbar.set_description("[converged] LP: {:.3f}".format(lp))
                    pbar.refresh()
                break
            else:
                patience -= 1
        else:
            patience = init_patience

    return np.array(log_probs)[1:], hmm, posterior
