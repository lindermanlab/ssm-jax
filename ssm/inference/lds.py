from collections import namedtuple

import jax.numpy as np
from jax import jit, value_and_grad, hessian, vmap, grad
from jax.scipy.linalg import solve_triangular
from jax import lax

from ssm.models.lds import GaussianLDS
from ssm.distributions import EXPFAM_DISTRIBUTIONS
from ssm.utils import Verbosity, ssm_pbar, sum_tuples

import jax.experimental.optimizers as optimizers

# TODO @slinderman @schlagercollin: Consider data classes
LDSPosterior = namedtuple(
    "LDSPosterior", ["marginal_likelihood",
                     "expected_states",
                     "expected_states_squared",
                     "expected_transitions"]
)


def lds_filter(J_diag, J_lower_diag, h, logc):
    seq_len, dim, _ = J_diag.shape

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = np.concatenate((J_lower_diag, np.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        # Predict -- Cholesky approach seems unstable!
        sqrt_Jc = np.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * np.log(2 * np.pi)
        log_Z += -np.sum(np.log(np.diag(sqrt_Jc)))
        log_Z += 0.5 * np.dot(trm1.T, trm1)
        Jp = -np.dot(trm2.T, trm2)
        hp = -np.dot(trm2.T, trm1)

        # Alternative predict step:
        # log_Z = 0.5 * dim * np.log(2 * np.pi)
        # log_Z += -0.5 * np.linalg.slogdet(Jc)[1]
        # log_Z += 0.5 * np.dot(hc, np.linalg.solve(Jc, hc))
        # Jp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, J_lower_diag_pad[t].T))
        # hp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, hc))

        new_carry = Jp, hp, lp + log_Z
        return new_carry, (Jc, hc)

    # Initialize
    Jp0 = np.zeros((dim, dim))
    hp0 = np.zeros((dim,))
    (_, _, lp), (filtered_Js, filtered_hs) = lax.scan(marginalize, (Jp0, hp0, logc), np.arange(seq_len))
    return lp, filtered_Js, filtered_hs


def lds_log_normalizer(J_diag, J_lower_diag, h, logc):
    lp, _, _ = lds_filter(J_diag, J_lower_diag, h, logc)
    return lp


# TODO @slinderman: finish porting this code!
def lds_sample(J_ini, h_ini, log_Z_ini,
               J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
               J_obs, h_obs, log_Z_obs):
    """
    Information form Kalman sampling for time-varying linear dynamical system with inputs.
    """
    T, D = h_obs.shape

    # Run the forward filter
    log_Z, filtered_Js, filtered_hs = \
        _kalman_info_filter(J_ini, h_ini, log_Z_ini,
                           J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
                           J_obs, h_obs, log_Z_obs)

    # Allocate output arrays
    samples = np.zeros((T, D))
    noise = npr.randn(T, D)

    # Initialize with samples of the last state
    samples[-1] = _sample_info_gaussian(filtered_Js[-1], filtered_hs[-1], noise[-1])

    # Run the Kalman information filter
    for t in range(T-2, -1, -1):
        # Extract blocks of the dynamics potentials
        J_11 = J_dyn_11[t] if J_dyn_11.shape[0] == T-1 else J_dyn_11[0]
        J_21 = J_dyn_21[t] if J_dyn_21.shape[0] == T-1 else J_dyn_21[0]
        h_1 = h_dyn_1[t] if h_dyn_1.shape[0] == T-1 else h_dyn_1[0]

        # Condition on the next observation
        J_post = filtered_Js[t] + J_11
        h_post = filtered_hs[t] + h_1 - np.dot(J_21.T, samples[t+1])
        samples[t] = _sample_info_gaussian(J_post, h_post, noise[t])

    return samples


# Expectation-Maximization
@jit
def lds_expected_states(J_diag, J_lower_diag, h, logc):
    """
    Retrieve the expected states for an LDS given its natural parameters.

    Args:
        J_diag:
        J_lower_diag:
        h:
        logc:

    Returns:
        marginal_likelihood (float): marginal likelihood of the data
        Ex
        ExxT
        ExnxT
    """

    f = value_and_grad(lds_log_normalizer, argnums=(0, 1, 2))
    marginal_likelihood, (E_neg_half_xxT, E_neg_xnxT, Ex) = f(J_diag, J_lower_diag, h, logc)

    ExxT = -2 * E_neg_half_xxT
    ExnxT = -E_neg_xnxT
    return marginal_likelihood, Ex, ExxT, ExnxT


# Expectation-Maximization
def _e_step(lds, data):
    marginal_likelihood, Ex, ExxT, ExnxT = lds_expected_states(*lds.natural_parameters(data))
    return LDSPosterior(marginal_likelihood, Ex, ExxT, ExnxT)


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
    J_21 = -1 * vmap(grad(grad(f, argnums=1), argnums=0)(states[:-1], states[1:])) # (dxtp1 dxt f)(states[:-1], states[1:])

    # observations
    f = lambda x, y: lds.emissions_distribution(x).log_prob(y)
    J_obs = -1 * vmap(hessian(f, argnums=0))(states, data)

    # combine into diagonal and lower diagonal blocks
    J_diag = J_obs
    J_diag = J_diag.at[0].add(J_init)
    J_diag = J_diag.at[:-1].add(J_11)
    J_diag = J_diag.at[1:].add(J_22)

    J_lower_diag = J_21
    
    # We have mu=states, J. Now we need h = J x
    # We know that J is block triagonal so we break up computation
    h_ini = J_init @ states[0]

    h_dyn_1 = (J_11 @ states[:-1][:, :, None])[:, :, 0]
    h_dyn_1 += (np.swapaxes(J_21, -1, -2) @ states[1:][:, :, None])[:, :, 0]

    h_dyn_2 = (J_22 @ states[1:][:, :, None])[:, :, 0]
    h_dyn_2 += (J_21 @ states[:-1][:, :, None])[:, :, 0]

    h_obs = (J_obs @ states[:, :, None])[:, :, 0]
    return h_ini, h_dyn_1, h_dyn_2, h_obs
    
    
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

    # Now convert x, J -> h = J x
    

    # Then run message passing with J and h to get E[x], E[xxT], ...

    # return LDSPosterior constructed from expectatiosn above


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
