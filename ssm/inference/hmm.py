"""
TODO: organize HMM-specific EM functions elsewhere
"""
from collections import namedtuple
from textwrap import dedent

import jax.numpy as np
from jax import jit, lax, value_and_grad

from functools import partial

import jax.scipy.special as spsp
import jax.random as npr
import jax.experimental.optimizers as optimizers

from ssm.models.hmm import HMM
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

# TODO @slinderman @schlagercollin: Consider data classes
HMMPosterior = namedtuple(
    "HMMPosterior", ["marginal_likelihood", "expected_states", "expected_transitions"]
)


### Core message passing routines
def hmm_log_normalizer(log_initial_state_probs, log_transition_matrix, log_likelihoods):
    """
    Compute the marginal likelihood (i.e. log normalizer) under
    a Hidden Markov Model (HMM) with the specified natural parameters.
    
    The normalizer is computed via the forward message passing recursion,

    .. math::
        \log \\alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \\alpha_{t,i} +
                                \log p(x_t | z_t = i) +
                                \log p(z_{t+1} = j | z_t = i) \}

    where, 
    
    .. math::
        \\alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).

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
hmm_expected_states.__doc__ = dedent(   
    """
    Compute posterior expectations of the latent states in a Hidden Markov
    Model (HMM) with the specified natural parameters. 
    
    The expectations are computed using a special property of exponential 
    family distributions: the expected sufficient statistics are equal to 
    the gradient of the log normalizer with respect to the natural parameters.

    For an HMM, the sufficient statistics are simply indicator functions

    .. math::

            t(z)_1 &= [\mathbb{I}[z_1 = 1], \ldots, \mathbb{I}[z_1=K]] \\\\
            t(z)_2 &= [\mathbb{I}[z_t = 1], \ldots, \mathbb{I}[z_t=K]] \\text{ for } t = 1, ..., T \\\\
            t(z)_3 &= [\mathbb{I}[z_t = k_1, z_{t+1} = k_2]] \\text{ for } t = 1, ..., T-1; k_1 = 1,...,K; k_2 = 1, ..., K
        
    The expected sufficient statistics are the probabilities of these
    events under the posterior distribution determined by the natural parameters,
    i.e. arguments to this function.

    Args:
        log_initial_distn: Log of the initial state distribution.
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
        The log normalizer and a tuple including of expected sufficient statistics.
    """
)

def hmm_expected_log_joint(log_initial_state_probs,
                           log_transition_matrix,
                           log_likelihoods,
                           posterior):
    """The expected log joint probability of an HMM given a posterior over the latent states.
    
    .. math::
        \mathbb{E}_{q(z)} \left[\log p(x, z, \\theta)\\right]
        
    where,
    
    .. math:: 
        q(z) = p(z|x, \\theta).
        
    Recall that this is the component of the ELBO that is dependent upon the parameters.
    
    .. math::
        \mathcal{L}(q, \\theta) = \mathbb{E}_{q(z)} \left[\log p(x, z, \\theta) - \log q(z) \\right]

    Args:
        log_initial_state_probs ([type]): [description]
        log_transition_matrix ([type]): [description]
        log_likelihoods ([type]): [description]
        posterior ([type]): [description]

    Returns:
        elp ([type]): expected log probability
    """
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
    log_probs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience
    for itr in pbar:
        hmm, posterior = step(hmm)
        lp = posterior.marginal_likelihood
        assert np.isfinite(lp), "NaNs in marginal log probability"
        log_probs.append(lp)
        if verbosity:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        # TODO: make this cleaner with patience
        if itr > 1 and abs(log_probs[-1] - log_probs[-2]) < tol:
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

def stochastic_em(hmm,
                  datas,
                  num_epochs=100,
                  verbosity=Verbosity.DEBUG,
                  key=npr.PRNGKey(0),
                  learning_rate=1e-3,
                ):
    """Stochastic EM implemented using mini-batch SGD on expected log-joint. 
    
    Note that this is implementation does not use the M-steps and convex-combinations of the
    expected sufficient statistics.

    Args:
        hmm ([type]): The HMM model to fit.
        datas ([type]): Observed data of the form ``(B, T, D)`` where ``B`` is a batch dimension
            indicating different trials of length T. Currently, all trials must be the same length.
        num_epochs (int, optional): Number of epochs to run stochastic EM. Defaults to 100.
        verbosity ([type], optional): Verbosity of output. Defaults to Verbosity.DEBUG.
        key ([type], optional): Random seed. Defaults to npr.PRNGKey(0).
        learning_rate ([type], optional): [description]. Defaults to 1e-3.

    Returns:
        lls ([type]): The expected log joint objective per iteration
        fitted_hmm ([HMM]): Output HMM model with fitted parameters.
    """
    
    assert len(datas.shape) == 3, "stochastic em should be used on data with a leading batch dimension"
    M = len(datas)
    T = sum([data.shape[0] for data in datas])
    
    perm = np.array([npr.permutation(k, M) for k in npr.split(key, num_epochs)])
    
    # import numpy as onp
    # perm = [onp.random.permutation(M) for _ in range(num_epochs)]
    
    def _get_minibatch(itr):
        epoch = itr // M
        m = itr % M
        i = perm[epoch][m]
        return datas[i]
    
    def _objective(new_hmm, curr_hmm, itr):
        
        # Grab a minibatch
        data = _get_minibatch(itr)
        Ti = data.shape[0]
        
        # E step (compute posterior using previous parameters)
        posterior = _e_step(curr_hmm, data)
        
        # Compute the expected log joint (component of ELBO dependent on parameters)
        log_initial_state_distn, log_transition_matrix, log_likelihoods = new_hmm.natural_parameters(data)
        
        obj = 0  # TODO prior
        obj += np.sum(posterior.expected_states[0] * log_initial_state_distn) * M
        obj += np.sum(posterior.expected_transitions * log_transition_matrix) * (T - M) / (Ti - 1)
        obj += np.sum(posterior.expected_states * log_likelihoods) * T / Ti
        return -obj / T
    
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(hmm)
    
    # @partial(jit, static_argnums=0)
    @jit
    def step(itr, opt_state, prev_hmm):
        value, grads = value_and_grad(_objective)(get_params(opt_state), prev_hmm, itr)
        prev_hmm = get_params(opt_state)
        opt_state = opt_update(itr, grads, opt_state)
        return value, opt_state, prev_hmm
    
    # Set up the progress bar
    pbar = ssm_pbar(num_epochs * M, verbosity, "Epoch {} Itr {} LP: {:.1f}", 0, 0, 0)

    # Run the optimizer
    prev_hmm = hmm
    lls = []
    for itr in pbar:
        value, opt_state, prev_hmm = step(itr, opt_state, prev_hmm)
        epoch = itr // M
        m = itr % M
        lls.append(-value * T)
        if verbosity:
            if itr % 10 == 0:  # update description every 10 iter to prevent warnings
                pbar.set_description(f"Epoch {epoch} Itr {m} LP: {lls[-1]:.1f}")
    fitted_hmm = get_params(opt_state)
    return lls, fitted_hmm
    