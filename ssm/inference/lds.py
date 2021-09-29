import jax.numpy as np
from jax import jit, value_and_grad, grad, hessian, vmap, jacfwd, jacrev
import jax.random as jr

from ssm.models.lds import GLMLDS, GaussianLDS
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import jax.experimental.optimizers as optimizers


# Expectation-Maximization
def _exact_e_step(lds, data):
    return MultivariateNormalBlockTridiag(*lds.natural_parameters(data))


def _exact_marginal_likelihood(lds, data, posterior=None):
    """
    For a Gaussian LDS, we can compute the exact marginal likelihood of
    the data (y) given the posterior p(x | y) via Bayes' rule:

        \log p(y) = \log p(y, x) - \log p(x | y)

    This equality holds for _any_ choice of x. We'll use the posterior mean.
    """
    if posterior is None:
        posterior = _exact_e_step(lds, data)
    states = posterior.mean
    return lds.log_probability(states, data) - posterior.log_prob(states)


def _exact_m_step_initial_distribution(lds, data, posterior, prior=None):
    expfam = EXPFAM_DISTRIBUTIONS[lds._initial_distribution.name]

    # Extract sufficient statistics
    Ex = posterior.mean[0]
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
    Ex = posterior.mean
    ExxT, ExnxT = posterior.second_moments

    # Sum over time
    sum_x = Ex[:-1].sum(axis=0)
    sum_y = Ex[1:].sum(axis=0)
    sum_xxT = ExxT[:-1].sum(axis=0)
    sum_yxT = ExnxT.sum(axis=0)
    sum_yyT = ExxT[1:].sum(axis=0)
    stats = (sum_x, sum_y, sum_xxT, sum_yxT, sum_yyT)
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
    Ex = posterior.mean
    ExxT, _ = posterior.second_moments

    # Sum over time
    sum_x = Ex.sum(axis=0)
    sum_y = data.sum(axis=0)
    sum_xxT = ExxT.sum(axis=0)
    sum_yxT = data.T.dot(Ex)
    sum_yyT = data.T.dot(data)
    stats = (sum_x, sum_y, sum_xxT, sum_yxT, sum_yyT)
    counts = len(data)

    if prior is not None:
        prior_stats, prior_counts = \
            expfam.prior_pseudo_obs_and_counts(prior.emissions_prior)
        stats = sum_tuples(stats, prior_stats)
        counts += prior_counts

    param_posterior = expfam.posterior_from_stats(stats, counts)
    return expfam.from_params(param_posterior.mode())


def _exact_m_step(lds, data, posterior, prior=None):
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
        patience: The number of steps to wait before algorithm convergence is declared.

    Returns:
        log_probs (array): log probabilities per iteration
        lds (LDS): the fitted LDS model
        posterior (LDSPosterior): the resulting posterior distribution object
    """

    @jit
    def step(lds):
        posterior = _exact_e_step(lds, data)
        lp = _exact_marginal_likelihood(lds, data, posterior=posterior)
        lds = _exact_m_step(lds, data, posterior)
        return lds, posterior, lp

    # Run the EM algorithm to convergence
    log_probs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience

    for itr in pbar:
        lds, posterior, lp = step(lds)
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


### Laplace EM for nonconjugate LDS with exponential family GLM emissions
def _laplace_find_mode(lds, x0, data, learning_rate=1e-3, num_iters=1500):
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


def _laplace_negative_hessian(lds, states, data):
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
    # TODO @collin: Check that this gives us the lower diagonal blocks and not the upper!
    J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0)(states[:-1], states[1:])) # (dxtp1 dxt f)(states[:-1], states[1:])

    # emissions
    f = lambda x, y: lds.emissions_distribution(x).log_prob(y)
    J_obs = -1 * vmap(hessian(f, argnums=0))(states, data)

    # combine into diagonal and lower diagonal blocks
    J_diag = J_obs
    J_diag = J_diag.at[0].add(J_init)
    J_diag = J_diag.at[:-1].add(J_11)
    J_diag = J_diag.at[1:].add(J_22)

    J_lower_diag = J_21

    return J_diag, J_lower_diag


def _laplace_e_step(lds, data, initial_states):
    """
    Laplace approximation to p(x | y, \theta) for non-Gaussian emission models.

    q <-- N(x*, -1 * J)
    J := H_{xx} \log p(y, x; \theta) |_{x=x*}
    """
    # find mode x*
    states = _laplace_find_mode(lds, initial_states, data)
    # compute negative hessian at the mode, J(x*)
    J_diag, J_lower_diag = _laplace_negative_hessian(lds, states, data)
    return MultivariateNormalBlockTridiag(J_diag,
                                          J_lower_diag,
                                          mean=states)


def _elbo(rng, model, data, posterior, num_samples=1):
    """
    For nonconjugate models like an LDS with GLM emissions, we can compute
    an _evidence lower bound_ (ELBO) using the joint probability and an
    approximate posterior q(x) \approx p(x | y):

        log p(y) \geq E_q[\log p(y, x) - \log q(x)]

    While in some cases the expectation can be computed in closed form, in
    general we will approximate it with ordinary Monte Carlo.
    """
    states = posterior.sample(rng, num_samples=num_samples)
    return np.mean(model.log_prob(states, data) - posterior.log_prob(states))


# def _elbo_alt(rng, model, data, posterior, num_samples=1):
#     states = posterior.sample(rng, num_samples=num_samples)
#     return np.mean(model.log_prob(states, data)) + posterior.entropy()


def _approx_m_step_emissions_distribution(rng, lds, data, posterior, prior=None, learning_rate=1e-3, num_timesteps=1000):
    x_sample = posterior.sample(rng)
    def _objective(_emissions_distribution):
        return _emissions_distribution(x_sample).log_prob(data)

    # do gradient descent on _emissions_distribution
    current_emissions_distribution = lds._emissions_distribution
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(current_emissions_distribution)

    # @jit
    def step(step, opt_state):
        grads = grad(_objective)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return opt_state

    for i in range(num_timesteps):
        opt_state = step(i, opt_state)

    new_emissions_distribution = get_params(opt_state)
    return new_emissions_distribution


def _laplace_m_step(rng, lds, data, posterior, prior=None):
    # TODO: M step for initial distribution needs a prior
    # initial_distribution = _exact_m_step_initial_distribution(lds, data, posterior, prior=prior)
    initial_distribution = lds._initial_distribution
    transition_distribution = _exact_m_step_dynamics_distribution(lds, data, posterior, prior=prior)
    emissions_distribution = _approx_m_step_emissions_distribution(rng, lds, data, posterior, prior=prior)
    return GLMLDS(initial_distribution,
                  transition_distribution,
                  emissions_distribution)


def laplace_em(
    rng,
    lds,
    data,
    num_iters: int=100,
    num_elbo_samples: int=1,
    tol: float=1e-4,
    verbosity: Verbosity=Verbosity.DEBUG,
    patience: int=5,
    ):
    """
    Run EM for an LDS with exponential family GLM emissions given observed data.

    Args:
        lds: The ExpfamLDS model to use perform EM over.
        data: A ``(B, T, D)`` or ``(T, D)`` data array.
        num_iters: Number of update iterations to perform EM.
        tol: Tolerance to determine EM convergence.
        verbosity: Determines whether a progress bar is displayed for the EM fit iterations.
        m_step_type: Determines how the model parameters are updated.
            Currently, only ``exact`` is supported for Gaussian LDS.
        patience: The number of steps to wait before algorithm convergence is declared.

    Returns:
        elbos (array): evidence lower bounds per iteration
        lds (LDS): the fitted ExpfamLDS model
        posterior (LDSPosterior): the resulting posterior distribution object
    """

    @jit
    def step(rng, lds, states):
        rng, elbo_rng, m_step_rng = jr.split(rng, 3)
        posterior = _laplace_e_step(lds, data, states)
        states = posterior.mean
        elbo = _elbo(elbo_rng, lds, data, posterior, num_samples=num_elbo_samples)
        lds = _laplace_m_step(m_step_rng, lds, data, posterior)
        return rng, lds, posterior, states, elbo

    # Run the Laplace EM algorithm to convergence
    elbos = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity:
        pbar.set_description("[jit compiling...]")
    init_patience = patience

    # Initialize the latent states to all zeros
    states = np.zeros((len(data), lds.latent_dim))

    for itr in pbar:
        rng, lds, posterior, states, elbo = step(rng, lds, states)
        elbos.append(elbo)
        if verbosity:
            pbar.set_description("ELBO: {:.3f}".format(elbo))

        # Check for convergence
        # TODO @collin: this is harder with Laplace EM since we have a Monte Carlo estimate of the ELBO
        if itr > 1 and abs(elbos[-1] - elbos[-2]) < tol:
            if patience == 0:
                if verbosity:
                    pbar.set_description("[converged] ELBO: {:.3f}".format(elbo))
                    pbar.refresh()
                break
            else:
                patience -= 1
        else:
            patience = init_patience

    return np.array(elbos), lds, posterior
