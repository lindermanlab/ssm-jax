"""
General EM routines
"""
import jax.numpy as np
from jax import jit,  value_and_grad, vmap
import jax.random as npr
import jax.experimental.optimizers as optimizers

from ssm.utils import Verbosity, format_dataset, ssm_pbar, sum_tuples


@format_dataset
def em(model,
       dataset,
       num_iters=100,
       tol=1e-4,
       verbosity=Verbosity.DEBUG,
    ):

    @jit
    def update(model):
        posteriors = model.e_step(dataset)
        lp = model.marginal_likelihood(dataset, posterior=posteriors).sum()
        model.m_step(dataset, posteriors)
        return model, posteriors, lp

    # Run the EM algorithm to convergence
    log_probs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)

    if verbosity > Verbosity.OFF:
        pbar.set_description("[jit compiling...]")

    for itr in pbar:
        model, posterior, lp = update(model)
        assert np.isfinite(lp), "NaNs in marginal log probability"

        log_probs.append(lp)
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}".format(lp))

        # Check for convergence
        if itr > 1:
            if log_probs[-1] < log_probs[-2]:
                print("Warning: LP is decreasing!")
                break

            if abs(log_probs[-1] - log_probs[-2]) < tol and verbosity > Verbosity.OFF:
                pbar.set_description("[converged] LP: {:.3f}".format(lp))
                pbar.refresh()
                break

    return np.array(log_probs), model, posterior


# TODO: adapt to class based approach
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