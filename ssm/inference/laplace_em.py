"""
Laplace EM (for non-conjugate LDS models such as GLM-LDS)
"""

import jax.numpy as np
import jax.random as jr
import jax.experimental.optimizers as optimizers
import jax.scipy.optimize
from jax import jit, value_and_grad, hessian, vmap, jacfwd, jacrev, lax

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import Verbosity, ssm_pbar


def _compute_laplace_mean(model, x0, data, method="L-BFGS", num_iters=50, learning_rate=1e-3):
    """Find the mode of the log joint probability for the Laplace approximation.

    Here, we seek to find

    .. math:
        \\argmax_{x_{1:T}} \log p(x_{1:T}, y_{1:T})

    where :math:`y_{1:T}` is fixed. It turns out this objective is a concave function.

    Args:
        ssm (SSM): The SSM model object. It must have continuous latent states.
        x0 (array, (num_timesteps, latent_dim)): Initial guess of state mode.
        data (array, (num_timesteps, obs_dim)): Observation data.
        method (str, optional): Optimization method to use. Choices are
            ["L-BFGS", "BFGS", "Adam"]. Defaults to "L-BFGS".
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

    if method == "BFGS" or "L-BFGS":
        # scipy minimize expects x to be shape (n,) so we flatten / unflatten
        def _objective(x_flattened):
            x = x_flattened.reshape(-1, dim)
            return -1 * np.sum(model.log_probability(x, data)) / scale

        optimize_results = jax.scipy.optimize.minimize(
            _objective,
            x0.ravel(),
            method="bfgs" if method == "BFGS" else "l-bfgs-experimental-do-not-rely-on-this",
            options=dict(maxiter=num_iters))

        # NOTE: optimize_results.status ==> 3 ("zoom failed") although it seems to be finding a max?
        x_mode = optimize_results.x.reshape(-1, dim)  # reshape back to (T, D)

    elif method == "Adam":

        _objective = lambda x: -1 * np.sum(model.log_probability(x, data)) / scale
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


def _compute_laplace_precision_blocks(model, states, data):
    """Get the negative Hessian at the given states for the Laplace approximation.

    Since we know the Hessian has a block tridiagonal structure, we can compute it in a piecewise
    fashion by taking the Hessian of the different components of the joint log probability of the SSM.

    Args:
        model (SSM): The SSM model object. It must have continuous latent states.
        states (array, (num_timesteps, latent_dim)): The states at which to evaluate the Hessian.
        data (array, (num_timesteps, obs_dim)): The observed data.

    Returns:
        J_diag (array, (num_timesteps, latent_dim, latent_dim)):
            The diagonal blocks of the tridiagonal negative Hessian.
        J_lower_diag (array, (num_timesteps - 1, latent_dim, latent_dim)):
            The lower diagonal blocks of the tridiagonal negative Hessian.
    """
    # initial distribution
    J_init = -1 * hessian(model.initial_distribution().log_prob)(states[0])

    # dynamics
    f = lambda xt, xtp1: model.dynamics_distribution(xt).log_prob(xtp1)
    J_11 = -1 * vmap(hessian(f, argnums=0))(states[:-1], states[1:])
    J_22 = -1 * vmap(hessian(f, argnums=1))(states[:-1], states[1:])
    J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0))(states[:-1], states[1:])

    # emissions
    f = lambda x, y: model.emissions_distribution(x).log_prob(y)
    J_obs = -1 * vmap(hessian(f, argnums=0))(states, data)

    # debug only if this flag is set
    if jax.config.jax_disable_jit:
        assert not np.any(np.isnan(J_init)), "nans in J_init"
        assert not np.any(np.isnan(J_11)), "nans in J_11"
        assert not np.any(np.isnan(J_22)), "nans in J_22"
        assert not np.any(np.isnan(J_21)), "nans in J_21"
        assert not np.any(np.isnan(J_obs)), "nans in J_obs"

    # combine into diagonal and lower diagonal blocks
    J_diag = J_obs
    J_diag = J_diag.at[0].add(J_init)
    J_diag = J_diag.at[:-1].add(J_11)
    J_diag = J_diag.at[1:].add(J_22)
    J_lower_diag = J_21
    return J_diag, J_lower_diag


# recognition network :: model, data --> approximate posterior



def laplace_approximation(model, data, initial_states, laplace_mode_fit_method="L-BFGS", num_laplace_mode_iters=10):
    """
    Laplace approximation to the posterior distribution for state space models
    with continuous latent states.
    """
    # Find the mean and precision of the Laplace approximation
    most_likely_states = _compute_laplace_mean(
        model, initial_states, data,
        num_iters=num_laplace_mode_iters,
        method=laplace_mode_fit_method)

    # The precision is given by the negative hessian at the mode
    J_diag, J_lower_diag = _compute_laplace_precision_blocks(
        model, most_likely_states, data)

    return MultivariateNormalBlockTridiag.infer_from_precision_and_mean(
        J_diag, J_lower_diag, most_likely_states)


def laplace_em(
    key,
    model,
    data,
    num_iters: int=100,
    num_elbo_samples: int=1,
    num_approx_m_iters: int=100,
    laplace_mode_fit_method: str="L-BFGS",
    num_laplace_mode_iters: int=100,
    tol: float=1e-4,
    verbosity: Verbosity=Verbosity.DEBUG,
    ):
    """Fit state space models such as an LDS with GLM emissions using Laplace EM.
    The state space models must have continuous latent states. Ideally, the
    log joint probability should also be concave so that a unique maximum exists.

    Laplace EM approximates the posterior as a Gaussian whose mean and covariance is
    set to match the mode and curvature (negative Hessian) of the posterior distribution.

    Note that because Laplace EM does not use the true posterior in the E-step, we are
    not guaranteed that the marginal log probability increases (as is true with exact EM).

    Args:
        rng (jax.random.PRNGKey): JAX random seed.
        ssm (SSM): The SSM model object to be fit.
        dataset (array, (num_timesteps, obs_dim)): The observed data.
        num_iters (int, optional): Number of iteration to run the Laplace EM algorithm. Defaults to 100.
        num_elbo_samples (int, optional): Number of Monte Carlo samples used to compute the ELBO expectation. Defaults to 1.
        laplace_mode_fit_method (str, optional): Optimization method used to compute the mode for the Laplace approximation.
            Must be one of ["L-BFGS", "BFGS", "Adam"]. Defaults to "L-BFGS".
        num_laplace_mode_iters (int, optional): Only relevant for when `laplace_mode_fit_method` is "Adam." Specifies the
            number of iterations to run the Adam updates. High values of iterations makes jit compilation slow. Defaults to 100.
        tol (float, optional): Tolerance to determine convergence of ELBO. Defaults to 1e-4.
        verbosity (Verbosity, optional): See Verbosity. Defaults to Verbosity.DEBUG.

    Returns:
        elbos (array, (num_iters,)): The ELBO objective per iteration. Ideally, this should increase as the model is fit.
        ssm (SSM): The fitted SSM object after running Laplace EM.
        posterior (MultivariateNormalBlockTridiag): The corresponding posterior distribution of the fitted SSM.
    """

    @jit
    def step(key, model, states):
        key, elbo_key, m_step_key = jr.split(key, 3)

        def _laplace_e_step_single(args):
            data, states = args
            posterior = laplace_approximation(model, data, states, laplace_mode_fit_method, num_laplace_mode_iters)
            return posterior

        # laplace e step
        posterior = lax.map(_laplace_e_step_single, (data, states))
        states = posterior.mean()

        # compute elbo
        elbos = model.elbo(elbo_key, data, posterior)
        elbo = np.mean(elbos)

        # m step (approx update for emissions)
        model.m_step(data, posterior, key=m_step_key)  # m step

        return key, model, posterior, states, elbo

    # Run the Laplace EM algorithm to convergence
    elbos = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity and not jax.config.jax_disable_jit:
        pbar.set_description("[jit compiling...]")

    # Initialize the latent states to all zeros
    states = np.zeros((data.shape[0], data.shape[1], model.latent_dim))

    for itr in pbar:
        key, model, posteriors, states, elbo = step(key, model, states)
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

    return np.array(elbos), model, posteriors
