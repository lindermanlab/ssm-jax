import jax
import jax.numpy as np
import jax.random as jr
from jax import jit, vmap
from ssm.utils import Verbosity, debug_rejit, ensure_has_batch_dim, ssm_pbar

import flax.optim as opt

import ssm.debug as debug
from ssm.debug import scan

@ensure_has_batch_dim(model_arg="model")
def deep_variational_inference(key,
             model,
             data,
             rec_net,
             posterior,
             learning_rate=1e-3,
             covariates=None,
             metadata=None,
             num_iters=100,
             tol=1e-4,
             verbosity=Verbosity.DEBUG,
             # Only learn the recognition network
             recognition_only=False,
             init_emissions_params=None,
             **kwargs
    ):

    assert(len(data.shape) == 3)
    batch_size, seq_len, data_dim = data.shape
    latent_dim = model.latent_dim

    rng1, rng2 = jr.split(key)

    def _update(key, rec_opt, dec_opt, model, posterior):
        def loss(network_params, posterior):
            rec_params, dec_params = network_params
            # We need the recognition networks to take care of vmapping
            potentials = rec_net.apply(rec_params, data)
            # These two methods have auto-batching
            posterior = posterior.update(model, data, potentials, covariates=covariates, metadata=metadata)
            
            if not recognition_only:
                # We have to pass in the params like this
                model.emissions_network.update_params(dec_params)

            elbo_key = jr.split(key, data.shape[0])
            bound = model.elbo(elbo_key, data, posterior, covariates=covariates, metadata=metadata, num_samples=10)
            return -np.sum(bound, axis=0), (model, posterior)
        
        results = \
            jax.value_and_grad(lambda params: loss(params, posterior), has_aux=True)((rec_opt.target, dec_opt.target))
        (neg_bound, (model, posterior)), (rec_grad, dec_grad) = results

        if not recognition_only:
            # Update the model!
            model = model.m_step(data, posterior, covariates=covariates, metadata=metadata)
        new_rec_opt = rec_opt.apply_gradient(rec_grad)
        new_dec_opt = dec_opt.apply_gradient(dec_grad)
        
        return new_rec_opt, new_dec_opt, model, posterior, -neg_bound

    DEBUG = debug.DEBUG
    AUTO_DEBUG = debug.AUTO_DEBUG

    x_single = np.ones((seq_len, data_dim))
    z_single = np.ones((latent_dim,))

    # Initialize the parameters and optimizers
    rec_params = rec_net.init(rng1, x_single)
    rec_opt = opt.Adam(learning_rate=learning_rate).create(rec_params)

    dec_net = model.emissions_network
    dec_params = init_emissions_params or dec_net.init(rng2, z_single)
    dec_opt = opt.Adam(learning_rate=learning_rate).create(dec_params)

    dec_net.update_params(dec_params)

    # Run the EM algorithm to convergence
    bounds = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)

    if not DEBUG:
        if AUTO_DEBUG:
            print("Auto-debug mode on, when the model crashes the last iteration will be re-run in debug mode.")
        if verbosity > Verbosity.OFF:
            pbar.set_description("[jit compiling...]")
        update = jit(_update)
    else:
        print("Debug mode on, all jit-compiled functions will be decomplied.")
        update = _update

    for itr in pbar:
        this_key, key = jr.split(key, 2)
        if (not DEBUG) and AUTO_DEBUG:
            try:
                rec_opt, dec_opt, model, posterior, bound = update(this_key, 
                                                           rec_opt, 
                                                           dec_opt, 
                                                           model, 
                                                           posterior)
            except:
                print(
"Some jit-compiled code crashed. Running the last iteration in debug mode.")
                with debug.Debug():
                    rec_opt, dec_opt, model, posterior, bound = _update(this_key, 
                                                           rec_opt, 
                                                           dec_opt, 
                                                           model, 
                                                           posterior)
        else:
            rec_opt, dec_opt, model, posterior, bound = update(this_key, 
                                                           rec_opt, 
                                                           dec_opt, 
                                                           model, 
                                                           posterior)
        
        assert np.isfinite(bound), "NaNs in log probability bound"

        bounds.append(bound)
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}".format(bound))

    model.emissions_network.update_params(dec_opt.target)
    return np.array(bounds), (model, (rec_net, rec_opt.target)), posterior