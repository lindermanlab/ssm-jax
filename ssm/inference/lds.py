import jax.numpy as np
from jax import jit, value_and_grad, grad, hessian, vmap, jacfwd, jacrev
import jax.random as jr
from jax.flatten_util import ravel_pytree

from ssm.models.lds import GLMLDS, GaussianLDS
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import jax.experimental.optimizers as optimizers
import tensorflow_probability.substrates.jax as tfp
import jax.scipy.optimize


# Expectation-Maximization
def _exact_e_step(lds, data):
    return MultivariateNormalBlockTridiag(*lds.natural_parameters(data))


def _exact_marginal_likelihood(lds, data, posterior=None):
    """The exact marginal likelihood of the observed data.

    For a Gaussian LDS, we can compute the exact marginal likelihood of
    the data (y) given the posterior p(x | y) via Bayes' rule:

    .. math::
        \log p(y) = \log p(y, x) - \log p(x | y)

    This equality holds for _any_ choice of x. We'll use the posterior mean.

    Args:
        - lds (LDS): The LDS model.
        - data (array, (num_timesteps, obs_dim)): The observed data.
        - posterior (MultivariateNormalBlockTridiag):
            The posterior distribution on the latent states. If None,
            the posterior is computed from the `lds` via message passing.
            Defaults to None.

    Returns:
        - lp (float): The marginal log likelihood of the data.
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
    ):
    """
    Run EM for a Gaussian LDS given observed data.

    Args:
        lds: The Gaussian LDS model to use perform EM over.
        data: A ``(B, T, D)`` or ``(T, D)`` data array.
        num_iters: Number of update iterations to perform EM.
        tol: Tolerance to determine EM convergence.
        verbosity: Determines whether a progress bar is displayed for the EM fit iterations.

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

    for itr in pbar:
        lds, posterior, lp = step(lds)
        log_probs.append(lp)
        if verbosity:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        if itr > 1 and abs(log_probs[-1] - log_probs[-2]) < tol:
            if verbosity:
                pbar.set_description("[converged] LP: {:.3f}".format(lp))
                pbar.refresh()
                break

    return np.array(log_probs), lds, posterior


### Laplace EM for nonconjugate LDS with exponential family GLM emissions
def _compute_laplace_mean(lds, x0, data, method="BFGS", num_iters=50, learning_rate=1e-3):
    """Find the mode of the log joint for the Laplace approximation.

    Here, we seek to find

    .. math:
        \\argmax_{x_{1:T}} \log p(x_{1:T}, y_{1:T})

    where :math:`y_{1:T}` is fixed. It turns out this objective is a concave function.

    Args:
        lds (LDS): The LDS model object.
        x0 (array, (num_timesteps, latent_dim)): Initial guess of state mode.
        data (array, (num_timesteps, obs_dim)): Observation data.
        method (str, optional): Optimization method to use. Choices are
            ["BFGS", "Adam"]. Defaults to "BFGS".
        learning_rate (float, optional): [description]. Defaults to 1e-3.
        num_iters (int, optional): Only used when optimization method is "Adam."
            Specifies the number of update iterations. Defaults to 50.

    Raises:
        ValueError: If the method is not one of "Adam" or "BFGS."

    Returns:
        x_mode (array, (num_timesteps, latent_dim)): The most likely states.
            Or the mode of the log joint probability holding data fixed.
    """

    scale = x0.size
    dim = x0.shape[-1]

    if method == "BFGS":
        # scipy minimize expects x to be shape (n,) so we flatten / unflatten
        def _objective(x_flattened):
            x = x_flattened.reshape(-1, dim)
            return -1 * np.sum(lds.log_probability(x, data)) / scale

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            x0.ravel(),
            method="BFGS",
            options=dict(maxiter=num_iters))

        # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
        x_mode = optimize_results.x.reshape(-1, dim)  # reshape back to (T, D)

    elif method == "Adam":

        _objective = lambda x: -1 * np.sum(lds.log_probability(x, data)) / scale
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_state = opt_init(x0)

        @jit
        def step(step, opt_state):
            value, grads = value_and_grad(_objective)(get_params(opt_state))
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(num_iters):
            value, opt_state = step(i, opt_state)
        x_mode = get_params(opt_state)

    else:
        raise ValueError(f"method = {method} is not recognized. Should be one of ['Adam', 'BFGS']")

    return x_mode


def _compute_laplace_precision_blocks(lds, states, data):
    """Get the negative Hessian of the LDS for the Laplace approximation evaluated at x = states.

    Since we know the Hessian has a block tridiagonal structure, we can compute it in a piecewise
    fashion by taking the Hessian of the different components of the joint log probability of the LDS.

    Args:
        lds (LDS): The LDS model object.
        states (array, (num_timesteps, latent_dim)): The states at which to evaluate the Hessian.
        data (array, (num_timesteps, obs_dim)): The observed data.

    Returns:
        J_diag (array, (num_timesteps, latent_dim, latent_dim)):
            The diagonal blocks of the tridiagonal negative Hessian.
        J_lower_diag (array, (num_timesteps - 1, latent_dim, latent_dim)):
            The lower diagonal blocks of the tridiagonal negative Hessian.
    """
    # initial distribution
    J_init = -1 * hessian(lds.initial_distribution().log_prob)(states[0])

    # dynamics
    f = lambda xt, xtp1: lds.dynamics_distribution(xt).log_prob(xtp1)
    J_11 = -1 * vmap(hessian(f, argnums=0))(states[:-1], states[1:])
    J_22 = -1 * vmap(hessian(f, argnums=1))(states[:-1], states[1:])
    J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0))(states[:-1], states[1:])

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


def _laplace_e_step(lds, data, initial_states, num_laplace_mode_iters=10, laplace_mode_fit_method="Adam"):
    """
    Laplace approximation to the posterior distribution for nonconjugate models.
    """
    # Find the mean and precision of the Laplace approximation
    most_likely_states = _compute_laplace_mean(
        lds, initial_states, data,
        num_iters=num_laplace_mode_iters,
        method=laplace_mode_fit_method)

    # The precision is given by the negative hessian at the mode
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(
        lds, most_likely_states, data)

    return MultivariateNormalBlockTridiag(J_diag,
                                          J_lower_diag,
                                          mean=most_likely_states)


def _elbo(rng, model, data, posterior, num_samples=1):
    """
    For nonconjugate models like an LDS with GLM emissions, we can compute
    an _evidence lower bound_ (ELBO) using the joint probability and an
    approximate posterior :math:`q(x) \\approx p(x | y)`:

    .. math:
        log p(y) \geq \mathbb{E}_q \left[\log p(y, x) - \log q(x) \\right]

    While in some cases the expectation can be computed in closed form, in
    general we will approximate it with ordinary Monte Carlo.
    """
    if num_samples == 1:
        states = posterior._sample(seed=rng)
    else:
        # TODO: implement cls._sample_n for mvn_block_tridiag
        raise NotImplementedError

    return np.mean(model.log_probability(states, data) - posterior.log_prob(states))


# def _elbo_alt(rng, model, data, posterior, num_samples=1):
#     states = posterior.sample(rng, num_samples=num_samples)
#     return np.mean(model.log_prob(states, data)) + posterior.entropy()


def _approx_m_step_emissions_distribution(rng, lds, data, posterior, prior=None):
    """Update the parameters of the emissions distribution via an approximate M step using samples from posterior.

    Uses BFGS to optimize the expected log probability of the emission via Monte Carlo estimate.

    For nonconjugate models like the GLM-LDS, we do not have a closed form expression for the objective nor solution
    to the M step parameter update for the emissions model. This is because the objective is technically an
    expectation under a Gaussian posterior on the latent states.

    We can approximate an update to our emissions distribution using a Monte Carlo estimate of this expectation,
    wherein we sample latent-state trajectories from our (potentially approximate) posterior and use these samples
    to compute the objective (the log probability of the data under the emissions distribution).

    Args:
        rng (jax.random.PRNGKey): JAX random seed.
        lds (LDS): The LDS model object.
        data (array, (num_timesteps, obs_dim)): Array of observed data.
        posterior (MultivariateNormalBlockTridiagonal):
            The LDS posterior object.
        prior (LDSPrior, optional): The prior distributions. Not yet supported. Defaults to None.

    Returns:
        emissions_distribution (tfp.distributions.Distribution):
            A new emissions distribution object with the updated parameters.
    """
    x_sample = posterior._sample(seed=rng)

    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    flat_emissions_distribution, unravel = ravel_pytree(lds._emissions_distribution)
    def _objective(flat_emissions_distribution):
        # TODO: Consider proximal gradient descent to counter sampling noise
        emissions_distribution = unravel(flat_emissions_distribution)
        return -1 * np.mean(emissions_distribution.predict(x_sample).log_prob(data))

    optimize_results = jax.scipy.optimize.minimize(
        _objective,
        flat_emissions_distribution,
        method="BFGS")

    # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
    return unravel(optimize_results.x)


def _laplace_m_step(rng, lds, data, posterior, prior=None, num_approx_m_iters=100):
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
    num_approx_m_iters: int=100,
    laplace_mode_fit_method: str="BFGS",
    num_laplace_mode_iters: int=100,
    tol: float=1e-4,
    verbosity: Verbosity=Verbosity.DEBUG,
    ):
    """Fit potentially nonconjugate LDS models such as LDS with GLM emissions using Laplace EM.

    Laplace EM approximates the posterior as a Gaussian whose mean and covariance is
    set to match the mode and curvature (negative Hessian) of the posterior distribution.

    Note that because Laplace EM does not use the true posterior in the E-step, we are
    not guaranteed that the marginal log probability increases (as is true with exact EM).

    Args:
        rng (jax.random.PRNGKey): JAX random seed.
        lds (LDS): The LDS model object to be fit.
        data (array, (num_timesteps, obs_dim)): The observed data.
        num_iters (int, optional): Number of iteration to run the Laplace EM algorithm. Defaults to 100.
        num_elbo_samples (int, optional): Number of Monte Carlo samples used to compute the ELBO expectation. Defaults to 1.
        laplace_mode_fit_method (str, optional): Optimization method used to compute the mode for the Laplace approximation.
            Must be one of ["BFGS", "Adam"]. Defaults to "BFGS".
        num_laplace_mode_iters (int, optional): Only relevant for when `laplace_mode_fit_method` is "Adam." Specifies the
            number of iterations to run the Adam updates. High values of iterations makes jit compilation slow. Defaults to 100.
        tol (float, optional): Tolerance to determine convergence of ELBO. Defaults to 1e-4.
        verbosity (Verbosity, optional): See Verbosity. Defaults to Verbosity.DEBUG.

    Returns:
        elbos (array, (num_iters,)): The ELBO objective per iteration. Ideally, this should increase as the model is fit.
        lds (LDS): The fitted LDS object after running Laplace EM.
        posterior (LDSPosterior): The corresponding posterior distribution of the fitted LDS.
    """

    @jit
    def step(rng, lds, states):
        rng, elbo_rng, m_step_rng = jr.split(rng, 3)
        posterior = _laplace_e_step(lds, data, states,
                                    laplace_mode_fit_method=laplace_mode_fit_method,
                                    num_laplace_mode_iters=num_laplace_mode_iters)
        states = posterior.mean
        elbo = _elbo(elbo_rng, lds, data, posterior, num_samples=num_elbo_samples)
        lds = _laplace_m_step(m_step_rng, lds, data, posterior, num_approx_m_iters=num_approx_m_iters)
        return rng, lds, posterior, states, elbo

    # Run the Laplace EM algorithm to convergence
    elbos = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity:
        pbar.set_description("[jit compiling...]")

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
            if verbosity:
                pbar.set_description("[converged] ELBO: {:.3f}".format(elbo))
                pbar.refresh()
                break

    return np.array(elbos), lds, posterior
