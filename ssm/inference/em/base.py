import jax.numpy as np

from ssm.inference.utils import Verbosity, sum_tuples, ssm_pbar


# TODO: create a general EM function here
def em(hmm, data, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG,
       m_step_type="exact", num_inner=1, patience=5,
):
    @jit
    def step(hmm):
        posterior = _e_step(hmm, data)
        if m_step_type == "exact":
            _exact_m_step(hmm, data, posterior)
        elif m_step_type == "sgd_marginal_likelihood":
            _generic_m_step(hmm, data, posterior, num_iters=num_inner)
        elif m_step_type == "sgd_expected_log_prob":
            _generic_m_step_elbo(hmm, data, posterior, num_iters=num_inner)
        else:
            raise ValueError("unrecognized")
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

    return np.array(log_probs)[1:], hmm, posterior