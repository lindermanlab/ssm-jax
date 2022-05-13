import jax
import jax.numpy as np
import jax.random as jr
from jax import jit, vmap
from ssm.utils import Verbosity, debug_rejit, ensure_has_batch_dim, ssm_pbar

import optax as opt
from flax.core import frozen_dict as fd

import ssm.debug as debug
from ssm.debug import scan, debug_jit

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
             elbo_samples=10,
             record_parameters=False,
             record_interval=10,
             **kwargs
    ):

    assert(len(data.shape) == 3)
    batch_size, seq_len, data_dim = data.shape
    latent_dim = model.latent_dim

    rng1, rng2 = jr.split(key)
    num_samples = elbo_samples
    print("Number of samples used for ELBO evaluation: {}".format(num_samples))

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
            bound = model.elbo(elbo_key, data, posterior, covariates=covariates, 
                metadata=metadata, num_samples=num_samples)
            return -np.sum(bound, axis=0), (model, posterior)
        
        results = \
            jax.value_and_grad(lambda params: loss(params, posterior), has_aux=True)((rec_opt[0], dec_opt[0]))
        (neg_bound, (model, posterior)), (rec_grad, dec_grad) = results

        if not recognition_only:
            # Update the model!
            model = model.m_step(data, posterior, covariates=covariates, metadata=metadata)

        updates, rec_opt_state = rec_optim.update(rec_grad, rec_opt[1])
        # Separate the gradients of the mean and the covariance and scale them separately
        # u = fd.unfreeze(updates)
        # cov = u["params"]["head_log_var_fn"]
        # mean = u["params"]["head_mean_fn"]
        # u["params"]["head_log_var_fn"] = jax.tree_map(lambda v: v * (1-grad_tradeoff), cov)
        # u["params"]["head_mean_fn"] = jax.tree_map(lambda v: v * grad_tradeoff, mean)
        # updates = fd.freeze(u)
        rec_params = opt.apply_updates(rec_opt[0], updates)
        updates, dec_opt_state = dec_optim.update(dec_grad, dec_opt[1])
        dec_params = opt.apply_updates(dec_opt[0], updates)
        
        return (rec_params, rec_opt_state), (dec_params, dec_opt_state), model, posterior, -neg_bound

    DEBUG = debug.DEBUG
    AUTO_DEBUG = debug.AUTO_DEBUG

    x_single = np.ones((seq_len, data_dim))
    z_single = np.ones((latent_dim,))

    if (kwargs.get("learning_rate_decay")):
        print("Using learning rate decay!")
        # Hard coded values, no tuning allowed >:(
        learning_rate = opt.exponential_decay(learning_rate, 100, 0.95)

    # Initialize the parameters and optimizers
    rec_params = rec_net.init(rng1, x_single)
    rec_optim = opt.adam(learning_rate=learning_rate)
    rec_opt_state = rec_optim.init(rec_params)

    dec_net = model.emissions_network
    dec_params = init_emissions_params or dec_net.init(rng2, z_single)
    dec_optim = opt.adam(learning_rate=learning_rate)
    dec_opt_state = dec_optim.init(dec_params)

    dec_net.update_params(dec_params)

    # Run the EM algorithm to convergence
    bounds = []
    pbar = ssm_pbar(num_iters, verbosity, "Iter {} LP: {:.3f}", 0, np.nan)

    if verbosity > Verbosity.OFF:
        pbar.set_description("[jit compiling...]")
    # New feature: the debug_jit wrapper!
    update = debug_jit(_update)

    past_rec_params = []

    rec_opt = (rec_params, rec_opt_state)
    dec_opt = (dec_params, dec_opt_state)

    grad_schedule = kwargs.get("grad_schedule") or (lambda _: 0.5)

    for itr in pbar:
        this_key, key = jr.split(key, 2)

        # r = grad_schedule(itr)

        rec_opt, dec_opt, model, posterior, bound = update(this_key, 
                                            rec_opt, dec_opt, model, posterior)
        
        assert np.isfinite(bound), "NaNs in log probability bound"

        bounds.append(bound)
        if record_parameters and itr % record_interval == 0:
            past_rec_params.append(rec_opt[0])
        
        if verbosity > Verbosity.OFF:
            pbar.set_description("LP: {:.3f}".format(bound))

    model.emissions_network.update_params(dec_opt[0])

    if record_parameters:
        past_rec_params.append(rec_opt[0])
        model_data = (model, (rec_net, past_rec_params))
    else:
        model_data = (model, (rec_net, rec_opt[0]))

    return np.array(bounds), model_data, posterior