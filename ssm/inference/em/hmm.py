from collections import namedtuple
import jax.numpy as np
import jax.random as jr
from jax import jit

from ssm.models.hmm.base import HMM
from ssm.inference.core import hmm_expected_states
from ssm.inference.priors import EXPFAM_DISTRIBUTIONS
from ssm.inference.utils import sum_tuples, Verbosity, ssm_pbar

HMMPosterior = namedtuple("HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"])


def _e_step(hmm, data):
    marginal_likelihood, (Ez0, Ezzp1, Ez) = \
        hmm_expected_states(*hmm.natural_parameters(data))
        
    return HMMPosterior(marginal_likelihood, Ez, Ezzp1)

def _exact_m_step_initial_distribution(hmm, data, posterior, prior=None):
    expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
    stats, counts = (posterior.expected_states[0],), 1

    if prior is not None:
        # Get stats from the prior
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.initial_prior)
    else:
        # Default to uniform prior (0 stats, 1 counts)
        prior_stats, prior_counts = (np.ones(hmm.num_states) + 1e-4,), 0
    stats = sum_tuples(stats, prior_stats)
    counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())

def _exact_m_step_transition_distribution(hmm, data, posterior, prior=None):
    expfam = EXPFAM_DISTRIBUTIONS["Categorical"]
    stats, counts = (posterior.expected_transitions,), 0

    if prior is not None:
        # Get stats from the prior
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.transition_prior)
    else:
        # Default to uniform prior (0 stats, 1 counts)
        prior_stats, prior_counts = (np.ones((hmm.num_states, hmm.num_states)) + 1e-4,), 0
    stats = sum_tuples(stats, prior_stats)
    counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())

def _exact_m_step_emissions_distribution(hmm, data, posterior, prior=None):
    # Use exponential family stuff for the emissions
    expfam = EXPFAM_DISTRIBUTIONS[hmm._emissions_distribution.name]
    stats = tuple(
        np.einsum('tk,t...->k...', posterior.expected_states, s) 
        for s in expfam.suff_stats(data))
    counts = np.sum(posterior.expected_states, axis=0)

    if prior is not None:
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())

def _exact_m_step(hmm, data, posterior, prior=None):
    initial_distribution = _exact_m_step_initial_distribution(hmm, data, posterior, prior=prior)
    transition_distribution = _exact_m_step_transition_distribution(hmm, data, posterior, prior=prior)
    emissions_distribution = _exact_m_step_emissions_distribution(hmm, data, posterior, prior=prior)
    return HMM(hmm.num_states,
               initial_distribution, 
               transition_distribution, 
               emissions_distribution)

def em(hmm, data, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG,
       m_step_type="exact", num_inner=1, patience=5,
):
    @jit
    def step(hmm):
        posterior = _e_step(hmm, data)
        if m_step_type == "exact":
            hmm = _exact_m_step(hmm, data, posterior)
        # elif m_step_type == "sgd_marginal_likelihood":
        #     _generic_m_step(hmm, data, posterior, num_iters=num_inner)
        # elif m_step_type == "sgd_expected_log_prob":
        #     _generic_m_step_elbo(hmm, data, posterior, num_iters=num_inner)
        # else:
        #     raise ValueError("unrecognized")
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

    return np.array(log_probs), hmm, posterior