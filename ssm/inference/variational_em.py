"""
Variational expectation maximization (vEM) routines
"""
import jax.numpy as np
import jax.random as jr
from jax import jit
from ssm.utils import Verbosity, ensure_has_batch_dim, ssm_pbar


@ensure_has_batch_dim(model_arg="model")
def variational_em(key,
                   model,
                   data,
                   posterior,
                   covariates=None,
                   metadata=None,
                   num_iters=100,
                   tol=1e-4,
                   verbosity=Verbosity.DEBUG,
                   callback=None
    ):
    """Fit a model using generalized EM. The E-step can be approximate, as long as the
    posterior output by the E step is compatible with the model's M-step.

    Assumes the model has the following methods for EM:
        - `model.m_step(data, posterior)`
        - `model.elbo(key, data, posterior)`

    and the posterior has the following method:
        - `posterior.update(model, data)` (i.e. variational E-step)

    Args:
        key (jr.PRNGKey): random number generator key used for stochastic m-steps and elbo calculations.
        model (ssm.base.SSM): the model to be fit
        data (PyTree): the observed data with leaf shape (B, T, ...).
        posterior (PosteriorDistribution)
        covariates (PyTree, optional): optional covariates with leaf shape (B, T, ...).
            Defaults to None.
        metadata (PyTree, optional): optional metadata with leaf shape (B, ...).
            Defaults to None.
        num_iters (int, optional): number of iterations of EM fit. Defaults to 100.
        tol (float, optional): tolerance in marginal lp to declare convergence. Defaults to 1e-4.
        verbosity (ssm.utils.Verbosity, optional): verbosity of fit. Defaults to Verbosity.DEBUG.
        callback: (function: ``model`` x ``posterior`` -> ``Any``, optional). If given, this
            function will be called after every EM iteration and its outputs will be returned
            as a list.

    Returns:
        bounds: log probabilities across EM iterations
        model: the fitted model
        posterior: the posterior over the inferred latent states

    If callback is not None, this function also returns:
        callback_outputs: list of callback outputs
    """
    @jit
    def update(key, model, posterior):
        k1, k2, k3 = jr.split(key, 3)
        model = model.m_step(data, posterior, covariates=covariates, metadata=metadata, key=k1)
        posterior = posterior.update(model, data, covariates=covariates, metadata=metadata, key=k2)
        bound = model.elbo(k3, data, posterior, covariates=covariates, metadata=metadata)
        callback_output = callback(model, posterior) if callback else None
        return model, posterior, bound, callback_output

    # Run the EM algorithm to convergence
    bounds = []
    callback_outputs = [callback(model, posterior) if callback else None]
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)

    if verbosity > Verbosity.OFF:
        pbar.set_description("[jit compiling...]")

    for itr in pbar:
        this_key, key = jr.split(key, 2)
        model, posterior, bound, callback_output = update(this_key, model, posterior)
        bounds.append(bound)
        callback_outputs.append(callback_output)

        assert np.isfinite(bound), "NaNs in log probability bound"
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}".format(bound))

        # Check for convergence
        if itr > 1:
            # TODO: Make a more general check
            if abs(bounds[-1] - bounds[-2]) < tol and verbosity > Verbosity.OFF:
                pbar.set_description("[converged] LP: {:.3f}".format(bound))
                pbar.refresh()
                break

    bounds = np.array(bounds)
    if callback is not None:
        return bounds, model, posterior, callback_outputs
    else:
        return bounds, model, posterior
