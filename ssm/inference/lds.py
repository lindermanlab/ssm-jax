from collections import namedtuple

import jax.numpy as np
from jax import jit, value_and_grad, hessian, vmap, jacfwd, jacrev
from jax.scipy.linalg import solve_triangular
from jax import lax

from ssm.models.lds import GaussianLDS
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import jax.experimental.optimizers as optimizers


def elbo(rng, model, data, posterior, num_samples=1):
    states = posterior.sample(rng, num_samples=num_samples)
    return np.mean(model.log_prob(states, data) - posterior.log_prob(states))


# def elbo2(rng, model, data, posterior, num_samples=1):
#     states = posterior.sample(rng, num_samples=num_samples)
#     return np.mean(model.log_prob(states, data)) + posterior.entropy()


# Expectation-Maximization
def _exact_e_step(gaussian_lds, data):
    return MultivariateNormalBlockTridiag(*gaussian_lds.natural_parameters)


def _fit_laplace_find_mode(lds, x0, data, learning_rate=1e-3, num_iters=1500):
    """Find the mode of the log joint for the Laplace approximation.

    Args:
        lds ([type]): [description]
        x0 ([type]): [description]
        data ([type]): [description]
        learning_rate ([type], optional): [description]. Defaults to 1e-3.

    Returns:
        [type]: [description]
    """

    scale = x0.size
    _objective = lambda x: -lds.log_probability(x, data)

    # TODO @collin: BFGS might be even better when _log_joint is concave
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(x0)

    @jit
    def step(step, opt_state):
        value, grads = value_and_grad(_objective)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    # guaranteed to convergence?
    for i in range(num_iters):
        value, opt_state = step(i, opt_state)

    return get_params(opt_state)

def _fit_laplace_negative_hessian(lds, states, data):
    """[summary]

    Args:
        lds ([type]): [description]
        states ([type]): [description]
        data ([type]): [description]

    Returns:
        J_diag
        J_lower_diag
    """

    # initial distribution
    J_init = -1 * hessian(lds.initial_distribution().log_prob)(states[0])

    # dynamics
    f = lambda xt, xtp1: lds.dynamics_distribution(xt).log_prob(xtp1)
    J_11 = -1 * vmap(hessian(f, argnums=0))(states[:-1], states[1:])
    J_22 = -1 * vmap(hessian(f, argnums=1))(states[:-1], states[1:])
    # J_21 = -1 * vmap(grad(grad(f, argnums=1), argnums=0)(states[:-1], states[1:])) # (dxtp1 dxt f)(states[:-1], states[1:])
    J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0)(states[:-1], states[1:])) # (dxtp1 dxt f)(states[:-1], states[1:])

    # observations
    f = lambda x, y: lds.emissions_distribution(x).log_prob(y)
    J_obs = -1 * vmap(hessian(f, argnums=0))(states, data)

    # combine into diagonal and lower diagonal blocks
    J_diag = J_obs
    J_diag = J_diag.at[0].add(J_init)
    J_diag = J_diag.at[:-1].add(J_11)
    J_diag = J_diag.at[1:].add(J_22)

    J_lower_diag = J_21

    return J_diag, J_lower_diag



def _laplace_e_step(lds, data):
    """
    Laplace approximation to p(x | y, \theta) for non-Gaussian emission models.

    q <-- N(x*, -1 * J)
    J := H_{xx} \log p(y, x; \theta) |_{x=x*}
    """

    # find mode x*
    states = _fit_laplace_find_mode(lds, x0, data)

    # compute negative hessian at the mode, J(x*)
    J_diag, J_lower_diag = _fit_laplace_negative_hessian(lds, states, data)

    return MultivariateNormalBlockTridiag(J_diag,
                                          J_lower_diag,
                                          mean=states)
    # # Now convert x, J -> h = J x

    # # Then run message passing with J and h to get E[x], E[xxT], ...
    # _, Ex, ExxT, ExnxT = lds_expected_states(J_diag, J_lower_diag, h, 0)
    # return LDSPosterior(np.nan, Ex, ExxT, ExnxT)


def _exact_m_step_initial_distribution(lds, data, posterior, prior=None):
    expfam = EXPFAM_DISTRIBUTIONS[lds._initial_distribution.name]

    # Extract sufficient statistics
    Ex = posterior.expected_states[0]
    ExxT = posterior.expected_states_squared[0]

    stats = (1.0, Ex, ExxT)
    counts = 1.0

    if prior is not None:
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.initial_prior)
        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())


def _exact_m_step_dynamics_distribution(lds, data, posterior, prior=None):
    expfam = EXPFAM_DISTRIBUTIONS[lds._dynamics_distribution.name]

    # Extract expected sufficient statistics from posterior
    Ex = posterior.expected_states[:-1].sum(axis=0)
    Ey = posterior.expected_states[1:].sum(axis=0)
    ExxT = posterior.expected_states_squared[:-1].sum(axis=0)
    EyxT = posterior.expected_transitions.sum(axis=0)
    EyyT = posterior.expected_states_squared[1:].sum(axis=0)
    stats = (Ex, Ey, ExxT, EyxT, EyyT)
    counts = len(data) - 1

    if prior is not None:
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.dynamics_prior)
        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())


def _exact_m_step_emissions_distribution(lds, data, posterior, prior=None):
    # Use exponential family stuff for the emissions
    expfam = EXPFAM_DISTRIBUTIONS[lds._emissions_distribution.name]

    # Extract expected sufficient statistics from posterior
    Ex = posterior.expected_states.sum(axis=0)
    Ey = data.sum(axis=0)
    ExxT = posterior.expected_states_squared.sum(axis=0)
    EyxT = data.T.dot(posterior.expected_states)
    EyyT = data.T.dot(data)
    stats = (Ex, Ey, ExxT, EyxT, EyyT)
    counts = len(data)

    if prior is not None:
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())


def _approx_m_step_emissions_distribution(lds, data, posterior, prior=None):

    # TODO: we need to make posterior an object with a sample method
    x_sample = posterior.sample(rng)

    def _objective(_emissions_distribution):
        return _emissions_distribution(x_sample).log_prob(data)

    # do gradient descent on _emissions_distribution

    # TODO: we need a GLM distribution object like the GaussianLinearRegression object



def _m_step(lds, data, posterior, prior=None):
    # TODO: M step for initial distribution needs a prior
    # initial_distribution = _exact_m_step_initial_distribution(lds, data, posterior, prior=prior)
    initial_distribution = lds._initial_distribution
    transition_distribution = _exact_m_step_dynamics_distribution(lds, data, posterior, prior=prior)
    emissions_distribution = _exact_m_step_emissions_distribution(lds, data, posterior, prior=prior)
    return GaussianLDS(initial_distribution,
                       transition_distribution,
                       emissions_distribution)


def em(lds,
       data,
       num_iters: int=100,
       tol: float=1e-4,
       verbosity: Verbosity=Verbosity.DEBUG,
       m_step_type: str="exact",
       patience: int=5,
    ):
    """
    Run EM for a Gaussian LDS given observed data.

    Args:
        lds: The Gaussian LDS model to use perform EM over.
        data: A ``(B, T, D)`` or ``(T, D)`` data array.
        num_iters: Number of update iterations to perform EM.
        tol: Tolerance to determine EM convergence.
        verbosity: Determines whether a progress bar is displayed for the EM fit iterations.
        m_step_type: Determines how the model parameters are updated.
            Currently, only ``exact`` is supported for Gaussian LDS.
        patience: The number of steps to wait before algorithm convergence is declared.

    Returns:
        log_probs (array): log probabilities per iteration
        lds (LDS): the fitted LDS model
        posterior (LDSPosterior): the resulting posterior distribution object
    """

    @jit
    def step(lds):
        posterior = _e_step(lds, data)
        if m_step_type == "exact":
            lds = _m_step(lds, data, posterior)

        # elif m_step_type == "sgd_marginal_likelihood":
        #     _generic_m_step(lds, data, posterior, num_iters=num_inner)
        # elif m_step_type == "sgd_expected_log_prob":
        #     _generic_m_step_elbo(lds, data, posterior, num_iters=num_inner)
        # else:
        #     raise ValueError("unrecognized")
        return lds, posterior

    # Run the EM algorithm to convergence
    log_probs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience

    for itr in pbar:
        lds, posterior = step(lds)
        lp = posterior.marginal_likelihood
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

    return np.array(log_probs), lds, posterior
