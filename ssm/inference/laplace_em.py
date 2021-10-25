"""
Laplace EM (for non-conjugate LDS models such as GLM-LDS)
"""

import jax.numpy as np
from jax import jit, value_and_grad, grad, hessian, vmap, jacfwd, jacrev
import jax.random as jr
from jax.flatten_util import ravel_pytree

from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import jax.experimental.optimizers as optimizers
import tensorflow_probability.substrates.jax as tfp
import jax.scipy.optimize

from functools import partial

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


def _laplace_e_step(lds, data, initial_states, laplace_mode_fit_method="BFGS", num_laplace_mode_iters=10, ):
    """
    Laplace approximation to the posterior distribution for nonconjugate LDS models.
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


def _elbo(model, rng, data, posterior, num_samples=1):
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
        # we can't seem to vmap over a function when lds is an argument
        # (even if we exclude lds from the vmap in_axes)
        # seems like it's an issue vmapping over tfp.dist objects (which are in lds' pytree)
        # a fix that I've found is to construct partial functions to fold-in the lds prior to
        # vmapping

        # this issue seems to be related to the general issue of using tfp distributions
        # with vmap (see https://github.com/tensorflow/probability/issues/1271)
        _laplace_e_step_partial = partial(_laplace_e_step, lds, 
                                          laplace_mode_fit_method=laplace_mode_fit_method,
                                          num_laplace_mode_iters=num_laplace_mode_iters)
        posteriors = vmap(_laplace_e_step_partial)(data, states)
        states = posteriors.mean
        elbo_rng = jr.split(elbo_rng, data.shape[0])
        _elbo_partial = partial(_elbo, lds, num_samples=num_elbo_samples)
        elbos = vmap(_elbo_partial)(elbo_rng, data, posteriors)
        elbo = np.mean(elbos)
        m_step_rng = jr.split(rng, data.shape[0])
        lds = lds.fit_with_posterior(data, posteriors, rng=m_step_rng)
        return rng, lds, posteriors, states, elbo

    # Run the Laplace EM algorithm to convergence
    elbos = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)
    if verbosity and not jax.config.jax_disable_jit:
        pbar.set_description("[jit compiling...]")

    # Initialize the latent states to all zeros
    states = np.zeros((data.shape[0], data.shape[1], lds.latent_dim))

    for itr in pbar:
        rng, lds, posteriors, states, elbo = step(rng, lds, states)
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

    return np.array(elbos), lds, posteriors