"""
General EM routines
"""
import warnings
import jax.numpy as np
from jax import jit, vmap
from ssm.utils import Verbosity, ensure_has_batch_dim, ssm_pbar, one_hot
from dataclasses import dataclass

@dataclass
class DummyPosterior:
    expected_states: np.ndarray

@jit # comment it out to debug or use id_print/id_tap
def update(model, data, fixed_zs, covariates, metadata, test_data, generic_m_step=False):
    if fixed_zs is not None:
        posterior = [DummyPosterior(one_hot(fixed_zs[i], model.num_states)) for i in range(len(fixed_zs))]
        lp = model.marginal_likelihood(data, None, covariates=covariates, metadata=metadata).sum()
    else:
        posterior = model.e_step(data, covariates=covariates, metadata=metadata)
        lp = model.marginal_likelihood(data, posterior, covariates=covariates, metadata=metadata).sum()
    if test_data is not None:
        test_posterior = model.e_step(test_data, covariates=covariates, metadata=metadata)
        test_lp = model.marginal_likelihood(test_data, test_posterior, 
                covariates=covariates, metadata=metadata).sum()
    else:
        test_lp = 0
    if not generic_m_step:
        model = model.m_step(data, posterior, covariates=covariates, metadata=metadata)
    else:
        model = model.generic_m_step(data, posterior, covariates=covariates, metadata=metadata)
    return model, posterior, lp, test_lp

#@ensure_has_batch_dim(model_arg="model")
def em(model,
       data,
       covariates=None,
       metadata=None,
       num_iters=100,
       tol=1e-4,
       verbosity=Verbosity.DEBUG,
       fixed_zs=None,
       test_data=None,
       callback=None,
       generic_m_step=False
    ):
    """Fit a model using EM.

    Assumes the model has the following methods for EM:

        - `model.e_step(data)` (i.e. E-step)
        - `model.m_step(dataset, posteriors)`
        - `model.marginal_likelihood(data, posterior)`

    Args:
        model (ssm.base.SSM): the model to be fit
        data (PyTree): the observed data with leaf shape (B, T, D).
        covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
            Defaults to None.
        metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
            Defaults to None.
        num_iters (int, optional): number of iterations of EM fit. Defaults to 100.
        tol (float, optional): tolerance in marginal lp to declare convergence. Defaults to 1e-4.
        verbosity (ssm.utils.Verbosity, optional): verbosity of fit. Defaults to Verbosity.DEBUG.

    Returns:
        log_probs: log probabilities across EM iterations
        model: the fitted model
        posterior: the posterior over the inferred latent states
    """

    # Run the EM algorithm to convergence
    log_probs = []
    test_log_probs = []
    callback_outputs = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}, test LP: {:.3f}", 0, np.nan, np.nan)

    if verbosity > Verbosity.OFF:
        pbar.set_description("[jit compiling...]")

    for itr in pbar:
        model, posterior, lp, test_lp = update(model, data, fixed_zs, covariates, metadata, test_data, generic_m_step) #update(model)
        callback_output = callback(model, posterior) if callback else None
        assert np.isfinite(lp), "NaNs in marginal log probability"

        log_probs.append(lp)
        test_log_probs.append(test_lp)
        callback_outputs.append(callback_output)
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}, test LP: {:.3f}".format(lp, test_lp))

        # Check for convergence
        if itr > 1:
            if log_probs[-1] < log_probs[-2]:
                pass # warnings.warn(UserWarning("LP is decreasing in EM fit!"))

            if abs(log_probs[-1] - log_probs[-2]) < tol and verbosity > Verbosity.OFF:
                pbar.set_description("[converged] LP: {:.3f}, test LP: {:.3f}".format(lp, test_lp))
                pbar.refresh()
                break

    return np.array(log_probs), model, posterior, np.array(test_log_probs), callback_outputs
