"""
SMC filtering/smoothing for SSMs.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

# Specific imports for here.
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from ssm.utils import Verbosity

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


# Define the standard plotting colours.
color_names = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple"
]


def smc(key,
        model,
        dataset,
        initialization_distribution=None,
        proposal=None,
        num_particles=50,
        resampling_criterion=None,
        resampling_function=None,
        use_stop_gradient_resampling=False,
        verbosity=default_verbosity):
    r"""Recover posterior over latent state given potentially batch of observation traces
    and a model.

    Assumes the model has the following methods:

        - `.initial_distribution(()`
        - `.dynamics_distribution(state)`
        - `.emissions_distribution(state)`

    which all return a TFP distribution.

    Assumes that the data and latent states are indexed 0:T-1, i.e. there is a latent
    state and observation at T=0 that exists prior to any dynamics.


    Args:
        key (JAX PRNG key, No size):
            JAX PRNG key.

        model (SSM object, No size):
            Defines the model.

        dataset (np.array, (batch x time x state_dim) --or-- (time x state_dim)):
            Data to condition on.  If the dataset has three dimensions then the leading dimension will be vmapped over.

        initialization_distribution (function, No size, default=model.initial_distribution):
            Allows a custom distribution to be used to propose the initial states from.  Using default value of
            None means that the prior is used as the proposal.
            Function takes arguments of (dataset, model, ...), but doesn't have to make use of them.  Allows
            more expressive initial proposals to be constructed.  The proposal must be a function that can be called
            as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may need to be 
            defined by closing over a proposal class as appropriate. 

        proposal (function, No size, default=model.dynamics_distribution):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, model, particles, time, p_dist, ...), but doesn't have to make
            use of
            them.
            Allows more expressive proposals to be constructed.  The proposal must be a function that can be called
            as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may need to be
            defined by closing over a proposal class as appropriate.

        num_particles (int, No size, default=50):
            Number of particles to use.

        resampling_criterion (function, No size, default=always_resample_criterion):
            Boolean function for whether to resample.

        resampling_function (function, No size, default=systematic_resampling):
            Resampling function.

        use_stop_gradient_resampling (bool, No size, default=False):
            Whether to use stop-gradient-resampling [Scibior & Wood, 2021].

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        particles (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Particles approximating the smoothing distribution.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        log_marginal_likelihood (np.array -- or -- np.float, (batch, ) -- or -- No size):
            Log-normalizer estimated via SMC.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        ancestry (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Full matrix of resampled ancestor indices.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        filtering_particles. (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Particles approximating the filtering distribution.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.
    """

    if resampling_criterion is None:
        resampling_criterion = always_resample_criterion

    if resampling_function is None:
        resampling_function = systematic_resampling

    # Close over the static arguments.
    single_smc_closed = lambda _k, _d: \
        _single_smc(_k, model, _d, initialization_distribution, proposal, num_particles,
                    resampling_criterion, resampling_function, use_stop_gradient_resampling, verbosity)

    # If there are three dimensions, it assumes that the dimensions correspond to
    # (batch_dim x time x states).  This copies the format of ssm->base->sample.
    # If there is a batch dimension, then we will vmap over the leading dim.
    if dataset.ndim == 3:
        key = jr.split(key, len(dataset))
        return vmap(single_smc_closed)(key, dataset)
    else:
        return single_smc_closed(key, dataset)


def _single_smc(key,
                model,
                dataset,
                initialization_distribution,
                proposal,
                num_particles,
                resampling_criterion,
                resampling_function,
                use_stop_gradient_resampling=False,
                verbosity=default_verbosity):
    r"""Recover posterior over latent state given a SINGLE dataset and model.

    Assumes the model has the following methods:

        - `.initial_distribution()`
        - `.dynamics_distribution(state)`
        - `.emissions_distribution(state)`

    which all return a TFP distribution.

    Assumes that the data and latent states are indexed 0:T-1, i.e. there is a latent
    state and observation at T=0 that exists prior to any dynamics.


    Args:
        key (JAX PRNG key, No size):
            JAX PRNG key.

        model (SSM object, No size):
            Defines the model.

        dataset (np.array, (time x state_dim)):
            Single dataset to condition on.

        initialization_distribution (function, No size, default=model.initial_distribution):
            Allows a custom distribution to be used to propose the initial states from.  Using default value of
            None means that the prior is used as the proposal.
            Function takes arguments of (dataset, model, ...), but doesn't have to make use of them.  Allows
            more expressive initial proposals to be constructed.  The proposal must be a function that can be called
            as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may need to be 
            defined by closing over a proposal class as appropriate. 

        proposal (function, No size, default=model.dynamics_distribution):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, parameters, particles, time, p_dist, ...), but doesn't have to make
            use of them.
            Allows more expressive proposals to be constructed.  The proposal must be a function that can be called
            as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may need to be
            defined by closing over a proposal class as appropriate.

        num_particles (int, No size, default=50):
            Number of particles to use.

        resampling_criterion (function, No size, default=always_resample_criterion):
            Boolean function for whether to resample.

        resampling_function (function, No size, default=systematic_resampling):
            Resampling function.

        use_stop_gradient_resampling (bool, No size, default=False):
            Whether to use stop-gradient-resampling [Scibior & Wood, 2021].

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        particles (np.array, (time x num_particles x state_dim)):
            Particles approximating the smoothing distribution.

        log_marginal_likelihood (np.float, No size):
            Log-normalizer estimated via SMC.

        ancestry (np.array, (time x num_particles x state_dim)):
            Full matrix of resampled ancestor indices.

        filtering_particles. (np.array, (time x num_particles x state_dim)):
            Particles approximating the filtering distribution.
    """

    # If no explicit proposal initial distribution is provided, default to prior.
    # The default initial implementation uses no inputs.
    # More complex proposals allow multiple arguments to be input and are
    # filtered on the inside.
    if initialization_distribution is None:
        initialization_distribution = lambda *args: model.initial_distribution()
    else:
        raise NotImplementedError()

    # If no explicit proposal is provided, default to BPF.
    # The default BPF implementation uses just the current particles as inputs.
    # More complex proposals allow multiple arguments to be input and are
    # filtered on the inside.
    if proposal is None:
        proposal = lambda *args: model.dynamics_distribution(args[2])

    # Do the forward pass.
    filtering_particles, log_marginal_likelihood, ancestry = \
        _smc_forward_pass(key,
                          model,
                          dataset,
                          initialization_distribution,
                          proposal,
                          num_particles,
                          resampling_criterion,
                          resampling_function,
                          use_stop_gradient_resampling,
                          verbosity)

    # Now do the backwards pass to generate the smoothing distribution.
    smoothing_particles = _smc_backward_pass(filtering_particles, ancestry, verbosity)

    return smoothing_particles, log_marginal_likelihood, ancestry, filtering_particles


def _smc_forward_pass(key,
                      model,
                      dataset,
                      initialization_distribution,
                      proposal,
                      num_particles,
                      resampling_criterion,
                      resampling_function,
                      use_stop_gradient_resampling=False,
                      verbosity=default_verbosity,):
    r"""Do the forward pass of an SMC sampler.


    Args:
        key (JAX PRNG key, No size):
            JAX PRNG key.

        model (SSM object, No size):
            Defines the model.

        dataset (np.array, (time x state_dim)):
            Single dataset to condition on.

        initialization_distribution (function, No size):
            Allows a custom distribution to be used to propose the initial states from.  Using default value of
            None means that the prior is used as the proposal.
            Function takes arguments of (dataset, model, ...), but doesn't have to make use of them.  Allows
            more expressive initial proposals to be constructed.  The proposal must be a function that can be called
            as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may need to be 
            defined by closing over a proposal class as appropriate. 

        proposal (function, No size):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, model, particles, time, p_dist, ...), but doesn't have to make
            use of them.  Allows more expressive proposals to be constructed.  The proposal must be a function that
            can be called as fn(...) and returns a TFP distribution object.  Therefore, the proposal function may
            need to be defined by closing over a proposal class as appropriate.

        num_particles (int, No size):
            Number of particles to use.

        resampling_criterion (function, No size):
            Boolean function for whether to resample.

        resampling_function (function, No size):
            Resampling function.

        use_stop_gradient_resampling (bool, No size, default=False):
            Whether to use stop-gradient-resampling [Scibior & Wood, 2021].

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        filtering_particles (np.array, (time x num_particles x state_dim)):
            Particles approximating the filtering distribution.

        log_marginal_likelihood (np.float, No size):
            Log-normalizer estimated via SMC.

        ancestors (np.array, (time x num_particles x state_dim)):
            Full matrix of resampled ancestor indices.
    """

    # Generate the initial distribution.
    initial_distribution = initialization_distribution(dataset, model)

    # Initialize the sweep using the initial distribution.
    key, subkey1, subkey2 = jr.split(key, num=3)
    initial_particles = initial_distribution.sample(seed=key, sample_shape=(num_particles, ), )

    # Resample particles under the zeroth observation.
    p_log_probability = model.initial_distribution().log_prob(initial_particles)
    q_log_probability = initial_distribution.log_prob(initial_particles)
    y_log_probability = model.emissions_distribution(initial_particles).log_prob(dataset[0])
    initial_incremental_log_weights = y_log_probability + p_log_probability - q_log_probability

    # Do an initial resampling step.
    initial_resampled_particles, _, accumulated_log_weights, initial_resampled = \
        do_resample(subkey2, initial_incremental_log_weights, initial_particles, resampling_criterion,
                    resampling_function, use_stop_gradient_resampling=use_stop_gradient_resampling)

    # # Use this to no do an initial resampling step.
    # initial_distribution = model.initial_distribution()
    # key, subkey1, subkey2 = jr.split(key, num=3)
    # initial_particles = initial_distribution.sample(seed=key, sample_shape=(num_particles, ), )
    # y_log_probability = model.emissions_distribution(initial_particles).log_prob(dataset[0])
    # initial_incremental_log_weights = y_log_probability
    # initial_resampled_particles = initial_particles
    # accumulated_log_weights = initial_incremental_log_weights
    # initial_resampled = False

    # Define the scan-compatible SMC iterate function.
    def smc_step(carry, t):
        key, particles, accumulated_log_weights = carry
        key, subkey1, subkey2 = jr.split(key, num=3)

        # Compute the p and q distributions.
        p_dist = model.dynamics_distribution(particles)
        q_dist = proposal(dataset, model, particles, t, p_dist)

        # Sample the new particles.
        new_particles = q_dist.sample(seed=subkey1)

        # Compute the incremental importance weight.
        p_log_probability = p_dist.log_prob(new_particles)
        q_log_probability = q_dist.log_prob(new_particles)
        y_log_probability = model.emissions_distribution(new_particles).log_prob(dataset[t])
        incremental_log_weights = p_log_probability - q_log_probability + y_log_probability

        # Update the log weights.
        accumulated_log_weights += incremental_log_weights

        # Resample particles.
        resampled_particles, ancestors, resampled_log_weights, should_resample = \
            do_resample(subkey2, accumulated_log_weights, new_particles, resampling_criterion,
                        resampling_function, use_stop_gradient_resampling=use_stop_gradient_resampling)

        return ((key, resampled_particles, resampled_log_weights),
                (resampled_particles, accumulated_log_weights, should_resample, ancestors))

    # Scan over the dataset.
    _, (filtering_particles, log_weights, resampled, ancestors) = jax.lax.scan(
        smc_step,
        (key, initial_resampled_particles, accumulated_log_weights),
        np.arange(1, len(dataset)))

    # Need to prepend the initial timestep.
    filtering_particles = np.concatenate((initial_resampled_particles[None, :], filtering_particles))
    log_weights = np.concatenate((initial_incremental_log_weights[None, :], log_weights))
    resampled = np.concatenate((np.asarray([initial_resampled]), resampled))
    ancestors = np.concatenate((np.arange(num_particles)[None, :], ancestors))

    # Average over particle dimension.
    p_hats = spsp.logsumexp(log_weights, axis=1) - np.log(num_particles)

    # Compute the log marginal likelihood.
    # Note that this needs to force the last accumulated incremental weight to be used.
    log_marginal_likelihood = np.sum(p_hats[:-1] * resampled[:-1]) + p_hats[-1]

    return filtering_particles, log_marginal_likelihood, ancestors


def _smc_backward_pass(filtering_particles,
                       ancestors,
                       verbosity=default_verbosity):
    r"""Do the backwards pass (pruning) given a SINGLE forward pass.

    TODO - Double check that the backwards pass code is correct.  I always get this wrong.

    Args:

        filtering_particles (np.array, (time x num_particles x state_dim)):
            Matrix of particles approximating *filtering* distribution.

        ancestors (np.array, (time x num_particles x state_dim)):
            Full matrix of ancestors in forward pass.

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    Returns:

        smoothing_particles (np.array):
            Matrix of particles approximating *smoothing* distribution.

    """

    def _backward_step(carry, t):
        ancestor = carry

        # Grab the ancestor state.
        next_smoothing_particles = jax.tree_map(lambda item: item[ancestor], filtering_particles[t-1])

        # Update the ancestor indices according to the resampling.
        next_ancestor = jax.tree_map(lambda item: item[ancestor], ancestors[t-1])

        return (next_ancestor, ), (next_smoothing_particles, )

    _, (smoothing_particles, ) = jax.lax.scan(
        _backward_step,
        (ancestors[-1], ),
        np.arange(1, len(filtering_particles)),
        reverse=True
    )

    # Append the final state to the return vector.
    smoothing_particles = np.concatenate((smoothing_particles, filtering_particles[-1][np.newaxis]))

    return smoothing_particles


def always_resample_criterion(unused_log_weights, unused_t):
    r"""A criterion that always resamples."""
    return True


def multinomial_resampling(key, log_weights, particles):
    r"""Resample particles with multinomial resampling.

    Args:

        key (JAX.PRNGKey, No size):
            A JAX PRNG key.

        log_weights (np.array, (num_particles x )):
            A [num_particles] ndarray, the log weights for each particle.

        particles (np.array, (num_particles x state_dim)):
            A pytree of [num_particles, ...] ndarrays that will be resampled.


    Returns:

        resampled_particles (np.array, (num_particles x state_dim)):
            A pytree of [num_particles, ...] ndarrays resampled via multinomial sampling.

    """
    num_particles = log_weights.shape[0]
    cat = tfd.Categorical(logits=log_weights)
    parents = cat.sample(sample_shape=(num_particles,), seed=key)
    return jax.tree_map(lambda item: item[parents], particles), parents


def systematic_resampling(key, log_weights, particles):
    r"""Resample particles with systematic resampling.

    Args:

        key (JAX.PRNGKey, No size):
            A JAX PRNG key.

        log_weights (np.array, (num_particles x )):
            A [num_particles] ndarray, the log weights for each particle.

        particles (np.array, (num_particles x state_dim)):
            A pytree of [num_particles, ...] ndarrays that will be resampled.


    Returns:

        resampled_particles (np.array, (num_particles x state_dim)):
            A pytree of [num_particles, ...] ndarrays resampled via systematic sampling.

    """

    # Grab the number of particles.
    num_particles = log_weights.shape[0]

    # Construct the bins.
    alpha = np.max(log_weights)
    zeroed_weights = np.exp(log_weights - alpha)
    normalized_weights = zeroed_weights / np.sum(zeroed_weights)
    cumsum_weights = np.cumsum(normalized_weights)
    bins_weights = np.concatenate((np.asarray([0]), cumsum_weights))

    # Force re-normalize.
    bins_weights = bins_weights / bins_weights[-1]

    # Construct the uniform vector.
    u = jr.uniform(key, minval=0, maxval=(1.0 / num_particles))
    uvec = u + np.linspace(0, (num_particles - 1) / num_particles, num_particles)

    # Use digitize to grab bin occupancies.
    # NOTE!  Finally found the problem, digitize treats everything to the left of
    # the zeroth bin index as the zeroth bin (which is kind of wild) so we have
    # to subtract one to make sure the indexing is correct as we never show it
    # values < 0.0.
    parents = np.digitize(uvec, bins_weights) - 1

    # # Do some error checking.
    # if np.min(parents) < 0:
    #     print('Lower limit error.')
    #     raise RuntimeError()
    #
    # if np.max(parents) >= num_particles:
    #     print('Upper limit error.')
    #     raise RuntimeError()

    # Do the resampling using treemap.
    return jax.tree_map(lambda item: item[parents], particles), parents


def do_resample(key,
                log_weights,
                particles,
                resampling_criterion=always_resample_criterion,
                resampling_function=systematic_resampling,
                num_particles=None,
                use_stop_gradient_resampling=False):
    r"""Do resampling.

    Allows a resampling condition to be passed in and evaluated to trigger adaptive resampling.

    If resampling occurs, then the returned incremental importance weights are zero.

    Args:
        key (JAX PRNG key, No size):
            JAX PRNG key.

        log_weights (np.array, (num_particles, ):
            Incremental importance weights of `particles`.

        particles (np.array, (num_particles x state_dim):
            Current particles.

        resampling_criterion (function, No size, default=always_resample_criterion):
            Boolean function for whether to resample.

        resampling_function (function, No size, default=systematic_resampling):
            Resampling function.

        num_particles (int, No size, default=len(log_weights)):
            Number of particles to resample.

        use_stop_gradient_resampling (bool, No size, default=False):
            Whether to use stop-gradient-resampling [Scibior & Wood, 2021].


    :return: Tuple:
        resampled_particles (np.array, (num_particles x state_dim):
            Resampled particles.

        ancestors (np.array, (num_particles, ):
            Ancestor indices of resampled particles.

        resampled_log_weights (num_particles, ):
            Incremental importance weights after any resampling -- resampling resets incremental importance weights.

        should_resample (bool, No size, default=False):
            Whether particles were resampled.

    """

    # If we have not specified a number of particles to resample, assume that we want
    # the particle set to remain the same size.
    if num_particles is None:
        num_particles = len(log_weights)

    should_resample = resampling_criterion(log_weights, 0)

    resampled_particles, ancestors = jax.lax.cond(
        should_resample,
        lambda args: resampling_function(*args),
        lambda args: (args[2], np.arange(num_particles)),
        (key, log_weights, particles)
    )

    resampled_log_weights = (1. - should_resample) * log_weights

    return resampled_particles, ancestors, resampled_log_weights, should_resample


def plot_single_sweep(particles, true_states, tag='', preprocessed=False):
    gen_label = lambda _k, _s: _s if _k == 0 else None

    if not preprocessed:
        single_sweep_median = jnp.median(particles, axis=1)
        single_sweep_lsd = jnp.quantile(particles, 0.17, axis=1)
        single_sweep_usd = jnp.quantile(particles, 0.83, axis=1)
    else:
        single_sweep_median = particles[0]
        single_sweep_lsd = particles[1]
        single_sweep_usd = particles[2]

    ts = jnp.arange(len(true_states))

    plt.figure(figsize=(10, 8))

    for _i, _c in zip(range(single_sweep_median.shape[1]), color_names):
        plt.plot(ts, single_sweep_median[:, _i], c=_c, label=gen_label(_i, 'Predicted'))
        plt.fill_between(ts, single_sweep_lsd[:, _i], single_sweep_usd[:, _i], color=_c, alpha=0.1)

        plt.plot(ts, true_states[:, _i], c=_c, linestyle='--', label=gen_label(_i, 'True'))

    plt.title(tag)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    # plt.xlim(-0.5, 3.5)
    plt.pause(0.1)
