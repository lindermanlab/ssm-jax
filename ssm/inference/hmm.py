"""
TODO: organize HMM-specific em functions elsewhere
"""
from collections import namedtuple

import jax.numpy as np
from jax import jit, lax, value_and_grad
import jax.scipy.special as spsp

from ssm.models.hmm import HMM
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

# TODO @slinderman @schlagercollin: Consider data classes
HMMPosterior = namedtuple(
    "HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"]
)


### Core message passing routines
def hmm_log_normalizer(log_initial_state_probs, log_transition_matrix, log_likelihoods):
    """Compute the marginal likelihood (i.e. log normalizer) under
    a Hidden Markov Model (HMM) with the specified natural parameters.
    The normalizer is computed via the forward message passing recursion,
        \log \alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \alpha_{t,i} +
                                \log p(x_t | z_t = i) +
                                \log p(z_{t+1} = j | z_t = i) \}
    where
        \alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).

    Args:
        log_initial_state_probs: Log of the initial state probabilities.
            A shape ``(K,)`` array where ``K`` is the number of states.

        log_transition_matrix: Log transition matrix or matrices. The shape
            must either be ``(K, K)`` where ``K`` is the number of states
            or ``(T-1, K, K)`` where ``T`` is the length of the sequence.
            In the former case, the entry ``[i,j]`` specifies the log
            probability of transitioning from state ``i`` to state ``j``.
            In the latter case, the ``[t,i,j]`` entry gives the log probability
            of transitioning from state ``z_t = i`` to state ``z_{t+1} = j``
            for ``t=1,...,T-1``.

        log_likelihoods: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.

    Returns:
        The log probability of the sequence, summing out the discrete states.
    """
    assert log_initial_state_probs.ndim == 1 and log_likelihoods.ndim == 2
    num_states = len(log_initial_state_probs)
    num_timesteps = len(log_likelihoods)
    assert log_likelihoods.shape[1] == num_states

    if log_transition_matrix.ndim == 2:
        # Stationary (fixed) transition probabilities
        assert log_transition_matrix.shape == (num_states, num_states)
        return _stationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    elif log_transition_matrix.ndim == 3:
        # Time-varying transition probabilities
        assert log_transition_matrix.shape == (
            num_timesteps - 1,
            num_states,
            num_states,
        )
        return _nonstationary_hmm_log_normalizer(
            log_initial_state_probs, log_transition_matrix, log_likelihoods
        )

    else:
        raise Exception("`log_transition_matrix` must be either 2d or 3d.")


def _stationary_hmm_log_normalizer(
    log_initial_state_probs, log_transition_matrix, log_likelihoods
):
    def marginalize(alpha, ll):
        alpha = spsp.logsumexp(alpha + ll + log_transition_matrix.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize, log_initial_state_probs, log_likelihoods[:-1]
    )

    # Note: alphas is interesting too, for forward filtering
    # Could return ..., np.row_stack((log_initial_state_probs, alphas))

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1])


def _nonstationary_hmm_log_normalizer(
    log_initial_state_probs, log_transition_matrices, log_likelihoods
):
    def marginalize(alpha, prms):
        log_P, ll = prms
        alpha = spsp.logsumexp(alpha + ll + log_P.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        marginalize,
        log_initial_state_probs,
        (log_transition_matrices, log_likelihoods[:-1]),
    )

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + log_likelihoods[-1])

hmm_expected_states = jit(value_and_grad(hmm_log_normalizer, argnums=(0, 1, 2)))

def hmm_expected_log_joint(log_initial_state_probs,
                           log_transition_matrix,
                           log_likelihoods,
                           posterior):

    elp = np.sum(posterior.expected_states[0] * log_initial_state_probs)
    elp += np.sum(posterior.expected_transitions * log_transition_matrix)
    elp += np.sum(posterior.expected_states * log_likelihoods)
    return elp


### EM algorithm
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

def em(hmm,
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