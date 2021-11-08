"""
General EM routines
"""
import warnings
import jax.numpy as np
from jax import jit, vmap
from ssm.utils import Verbosity, format_dataset, ssm_pbar


@format_dataset
def em(model,
       dataset,
       num_iters=100,
       tol=1e-4,
       verbosity=Verbosity.DEBUG,
    ):
    """Fit a model using EM.
    
    Assumes the model has the following methods for EM:
    
        - `.infer_posterior(data)` (i.e. E-step)
        - `.marginal_likelihood(data, posterior)`
        - `.m_step(dataset, posteriors)`

    Args:
        model (ssm.base.SSM): the model to be fit
        dataset ([np.ndarray]): the dataset with shape (B, T, D).
        num_iters (int, optional): number of iterations of EM fit. Defaults to 100.
        tol (float, optional): tolerance in marginal lp to declare convergence. Defaults to 1e-4.
        verbosity (ssm.utils.Verbosity, optional): verbosity of fit. Defaults to Verbosity.DEBUG.

    Returns:
        log_probs: log probabilities across EM iterations
        model: the fitted model
        posterior: the posterior over the inferred latent states
    """

    @jit
    def update(model):
        posteriors = vmap(model.infer_posterior)(dataset)
        lp = vmap(model.marginal_likelihood)(dataset, posteriors).sum()
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
                pass # warnings.warn(UserWarning("LP is decreasing in EM fit!"))

            if abs(log_probs[-1] - log_probs[-2]) < tol and verbosity > Verbosity.OFF:
                pbar.set_description("[converged] LP: {:.3f}".format(lp))
                pbar.refresh()
                break

    return np.array(log_probs), model, posterior
