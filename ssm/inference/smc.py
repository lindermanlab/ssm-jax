"""
SMC filtering/smoothing for SSMs.
"""
import jax
import warnings
import jax.numpy as jnp
from jax import jit, vmap
from ssm.utils import Verbosity, format_dataset, ssm_pbar

# Specific imports for here.
import jax.scipy as jscipy
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy as dc
import matplotlib.pyplot as plt

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def always_resample_criterion(unused_log_weights, unused_t):
    r"""A criterion that always resamples."""
    return True


def multinomial_resampling(key, log_weights, particles):
    r"""Resample particles with multinomial resampling.

    Args:
      key: A JAX PRNG key.
      log_weights: A [num_particles] ndarray, the log weights for each particle.
      particles: A pytree of [num_particles, ...] ndarrays that
        will be resampled.
    Returns:
      resampled_particles: A pytree of [num_particles, ...] ndarrays resampled via
        multinomial sampling.
    """
    num_particles = log_weights.shape[0]
    cat = tfd.Categorical(logits=log_weights)
    parents = cat.sample(sample_shape=(num_particles,), seed=key)
    return jax.tree_map(lambda item: item[parents], particles), parents


def systematic_resampling(key, log_weights, particles):
    r"""Resample particles with systematic resampling.

    Args:
      key: A JAX PRNG key.
      log_weights: A [num_particles] ndarray, the log weights for each particle.
      particles: A pytree of [num_particles, ...] ndarrays that
        will be resampled.
    Returns:
      resampled_particles: A pytree of [num_particles, ...] ndarrays resampled via
        systematic sampling.
    """

    # Grab the number of particles.
    num_particles = log_weights.shape[0]

    # Construct the bins.
    alpha = jnp.max(log_weights)
    zeroed_weights = jnp.exp(log_weights - alpha)
    normalized_weights = zeroed_weights / jnp.sum(zeroed_weights)
    cumsum_weights = jnp.cumsum(normalized_weights)
    bins_weights = jnp.concatenate((jnp.asarray([0]), cumsum_weights))

    # Force re-normalize.
    bins_weights = bins_weights / bins_weights[-1]

    # Construct the uniform vector.
    u = jax.random.uniform(key, minval=0, maxval=(1.0 / num_particles))
    uvec = u + jnp.linspace(0, (num_particles - 1) / num_particles, num_particles)

    # Use digitize to grab bin occupancies.
    # NOTE!  Finally found the problem, digitize treats everything to the left of
    # the zeroth bin index as the zeroth bin (which is kind of wild) so we have
    # to subtract one to make sure the indexing is correct as we never show it
    # values < 0.0.
    parents = jnp.digitize(uvec, bins_weights) - 1

    # # Do some error checking.
    # if jnp.min(parents) < 0:
    #     print('Lower limit error.')
    #     raise RuntimeError()
    #
    # if jnp.max(parents) >= num_particles:
    #     print('Upper limit error.')
    #     raise RuntimeError()

    # Do the resampling using treemap.
    return jax.tree_map(lambda item: item[parents], particles), parents


def do_resample(key,
                log_ws,
                particles,
                resampling_criterion=always_resample_criterion,
                resampling_fn=systematic_resampling,
                num_particles=None,
                use_sgr=False):
    r"""Do resampling.

    Allows a resampling condition to be passed in and evaluated to trigger adaptive resampling.

    If resampling occurs, then the returned incremental importance weights are zero.

    Args:
        key (JAX PRNG key):                 JAX PRNG key.
        log_ws (jnp.array):                 Incremental importance weights of `particles`.
        particles (jnp.array):              Current particles.
        resampling_criterion (fn):          Boolean function for whether to resample.
        resampling_fn (fn):                 Resampling operation.
        num_particles (int):                Number of particles to resample (defaults to len(particles)).
        use_sgr (bool):                     Whether to use stop-gradient-resampling [Scibior & Wood, 2021].

    :return: Tuple:
        resampled_particles (jnp.array):    Resampled particles.
        ancestors (jnp.array):              Ancestor indices of resampled particles.
        resampled_log_ws (jnp.array):       Incremental importance weights after any resampling.
        should_resample (bool):             Whether particles were resampled.
    """

    # If we have not specified a number of particles to resample, assume that we want
    # the particle set to remain the same size.
    if num_particles is None:
        num_particles = len(log_ws)

    should_resample = resampling_criterion(log_ws, 0)

    resampled_particles, ancestors = jax.lax.cond(
        should_resample,
        lambda args: resampling_fn(*args),
        lambda args: (args[2], jnp.arange(num_particles)),
        (key, log_ws, particles)
    )

    resampled_log_ws = (1. - should_resample) * log_ws

    return resampled_particles, ancestors, resampled_log_ws, should_resample


def smc(key,
        model,
        dataset,
        initialization_distribution=None,
        proposal=None,
        num_particles=50,
        resampling_criterion=always_resample_criterion,
        resampling_fn=systematic_resampling,
        use_sgr=False,
        verbosity=default_verbosity):
    r"""Recover posterior over latent state given potentially batch of observation traces
    and a model.

    Assumes the model has the following methods:

        - `.initial_distribution()`
        - `.dynamics_distribution(state)`
        - `.emissions_distribution(state)`

    which all return a TFP distribution.

    Assumes that the data and latent states are indexed 0:T-1, i.e. there is a latent
    state and observation at T=0 that exists prior to any dynamics.


    Args:
        key (JAX PRNG key):                     JAX PRNG key.
        model (SSM object):                     Defines the model.
        dataset (jnp.array):                    Data to condition on.  If the dataset has three
            dimensions then the leading dimension will be vmapped over.
        initialization_distribution (??, optional): Allows a custom distribution to be used to
            propose the initial states from.  Using default value of None means that the prior
            is used as the proposal.
        proposal (??, option):                  Allows a custom proposal to be used to propose
            transitions from.  Using default value of None means that the prior
            is used as the proposal.
        num_particles (int):                    Number of particles to use.
        resampling_criterion (fn):              Boolean function for whether to resample.
        resampling_fn (fn):                     Resampling operation.
        use_sgr (bool):                     Whether to use stop-gradient-resampling [Scibior & Wood, 2021].
        verbosity (??):                         Level of text output.


    :return: Tuple:
        particles (jnp.array):                  Resampled particles - either filtered or smoothed.
        log_marginal_likelihood (jnp.float):    Log-normalizer estimated via SMC.
        ancestry (jnp.array):                   Full matrix of resampled ancestor indices.
        filtering_particles.
    """

    # Close over the static arguments.
    single_smc_closed = lambda _k, _d: \
        _single_smc(_k, model, _d, initialization_distribution, proposal, num_particles,
                    resampling_criterion, resampling_fn, use_sgr, verbosity)

    # If there are three dimensions, it assumes that the dimensions correspond to
    # (batch_dim x time x states).  This copies the format of ssm->base->sample.
    # If there is a batch dimension, then we will vmap over the leading dim.
    if dataset.ndim == 3:
        key = jax.random.split(key, len(dataset))
        return jax.vmap(single_smc_closed)(key, dataset)
    else:
        return single_smc_closed(key, dataset)


def _single_smc(key,
                model,
                dataset,
                initialization_distribution,
                proposal,
                num_particles,
                resampling_criterion,
                resampling_fn,
                use_sgr=False,
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
        key (JAX PRNG key):                     JAX PRNG key.
        model (SSM object):                     Defines the model.
        dataset (jnp.array):                    Single data to condition on.
        initialization_distribution (??, optional): Allows a custom distribution to be used to
            propose the initial states from.  Using default value of None means that the prior
            is used as the proposal.
        proposal (??, option):                  Allows a custom proposal to be used to propose
            transitions from.  Using default value of None means that the prior
            is used as the proposal.
        num_particles (int):                    Number of particles to use.
        smooth (bool):                          Return particles from the filtering distribution or
            the smoothing distribution?
        resampling_criterion (fn):              Boolean function for whether to resample.
        resampling_fn (fn):                     Resampling operation.
        use_sgr (bool):                     Whether to use stop-gradient-resampling [Scibior & Wood, 2021].
        verbosity (??):                         Level of text output.


    :return: Tuple:
        particles (jnp.array):                  Resampled particles - either filtered or smoothed.
        log_marginal_likelihood (jnp.float):    Log-normalizer estimated via SMC.
        ancestry (jnp.array):                   Full matrix of resampled ancestor indices.
        filtering_particles.
    """

    # If no explicit proposal initial distribution is provided, default to prior.
    # The default initial implementation uses no inputs.
    # More complex proposals allow multiple arguments to be input and are
    # filtered on the inside.
    if initialization_distribution is None:
        initialization_distribution = lambda *args: model.initial_distribution()

    # If no explicit proposal is provided, default to BPF.
    # The default BPF implementation uses just the current particles as inputs.
    # More complex proposals allow multiple arguments to be input and are
    # filtered on the inside.
    if proposal is None:
        proposal = lambda *args: model.dynamics_distribution(args[0])

    # Do the forward pass.
    filtering_particles, log_marginal_likelihood, ancestry = \
        _smc_forward_pass(key,
                          model,
                          dataset,
                          initialization_distribution,
                          proposal,
                          num_particles,
                          resampling_criterion,
                          resampling_fn,
                          use_sgr,
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
                      resampling_fn,
                      use_sgr=False,
                      verbosity=default_verbosity,):
    r"""Do the forward pass of an SMC sampler.


    Args:
        key (JAX PRNG key):                 JAX PRNG key.
        model (SSM object):                 Defines the model.
        dataset (jnp.array):                Single dataset to condition on.
        initialization_distribution (??, optional): Allows a custom distribution to be used to
            propose the initial states from.  Using default value of None means that the prior
            is used as the proposal.
        proposal (??, optional):            Allows a custom proposal to be used to propose
            transitions from.  Using default value of None means that the prior
            is used as the proposal.
        num_particles (int):                Number of particles to use.
        resampling_criterion (fn):          Boolean function for whether to resample.
        resampling_fn (fn):                 Resampling operation.
        verbosity (??):                     Level of text output.


    :return: Tuple:
        filtering_particles (jnp.array):    Matrix of particles approximating *filtering* distribution.
        log_marginal_likelihood (float):    Estimate of the log normalizer computed by SMC.
        ancestors (jnp.array):              Full matrix of ancestors in forward pass.
    """

    # Generate the initial distribution.
    initial_dist = initialization_distribution(dataset, )

    # Initialize the sweep using the initial distribution.
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    initial_particles = initial_dist.sample(seed=key, sample_shape=(num_particles, ), )

    # Resample particles under the zeroth observation.
    p_log_prob = model.initial_distribution().log_prob(initial_particles)
    q_log_prob = initial_dist.log_prob(initial_particles)
    x_log_prob = model.emissions_distribution(initial_particles).log_prob(dataset[0])
    initial_incr_log_ws = p_log_prob - q_log_prob + x_log_prob

    # Do an initial resampling step.
    initial_resampled_particles, _, accumulated_log_ws, initial_resampled = \
        do_resample(subkey2, initial_incr_log_ws, initial_particles, resampling_criterion, resampling_fn)

    # # Uncomment this to force not doing an initial resampling step.
    # initial_resampled_particles = initial_particles
    # accumulated_log_ws = initial_incr_log_ws
    # initial_resampled = False

    # Define the scan-compatible SMC iterate function.
    def smc_step(carry, t):
        key, particles, accumulated_log_ws = carry
        key, subkey1, subkey2 = jax.random.split(key, num=3)

        # Propagate the particle particles.
        q_dist = proposal(particles, dataset, t)

        # Sample the new particles.
        new_particles = q_dist.sample(seed=subkey1)

        # Compute the incremental importance weight.
        p_log_prob = model.dynamics_distribution(particles).log_prob(new_particles)
        q_log_prob = q_dist.log_prob(new_particles)
        x_log_prob = model.emissions_distribution(new_particles).log_prob(dataset[t])
        incr_log_ws = p_log_prob - q_log_prob + x_log_prob

        # Update the log weights.
        accumulated_log_ws += incr_log_ws

        # Resample particles.
        resampled_particles, ancestors, resampled_log_ws, should_resample = \
            do_resample(subkey2, accumulated_log_ws, new_particles, resampling_criterion, resampling_fn, use_sgr)

        return ((key, resampled_particles, resampled_log_ws),
                (resampled_particles, accumulated_log_ws, should_resample, ancestors))

    # Scan over the dataset.
    _, (filtering_particles, log_weights, resampled, ancestors) = jax.lax.scan(
        smc_step,
        (key, initial_resampled_particles, accumulated_log_ws),
        jnp.arange(1, len(dataset)))

    # Need to prepend the initial timestep.
    filtering_particles = jnp.concatenate((initial_resampled_particles[None, :], filtering_particles))
    log_weights = jnp.concatenate((initial_incr_log_ws[None, :], log_weights))
    resampled = jnp.concatenate((jnp.asarray([initial_resampled]), resampled))

    # Average over particle dimension.
    p_hats = jscipy.special.logsumexp(log_weights, axis=1) - jnp.log(num_particles)

    # Compute the log marginal likelihood.
    # Note that this needs to force the last accumulated incremental weight to be used.
    log_marginal_likelihood = jnp.sum(p_hats[:-1] * resampled[:-1]) + p_hats[-1]

    return filtering_particles, log_marginal_likelihood, ancestors


def _smc_backward_pass(filtering_particles,
                       ancestors,
                       verbosity=default_verbosity):
    r"""Do the backwards pass (pruning) given a forward pass.

    TODO - Double check that the backwards pass code is correct.  I always get this wrong.

    Args:
        filtering_particles (jnp.array):        Matrix of particles approximating *filtering* distribution.
        ancestors (jnp.array):                  Full matrix of ancestors in forward pass.
        verbosity (??):                         Level of textual output.
    :return: smoothing_particles (jnp.array):   Matrix of particles approximating *smoothing* distribution.
    """

    def _backward_step(carry, t):
        next_an = carry

        # Grab the ancestor state.
        next_sp = jax.tree_map(lambda item: item[next_an], filtering_particles[t-1])

        # Update the ancestor indices according to the resampling.
        next_an = jax.tree_map(lambda item: item[next_an], ancestors[t-1])

        return (next_an, ), (next_sp, )

    _, (smoothing_particles, ) = jax.lax.scan(
        _backward_step,
        (ancestors[-1], ),
        jnp.arange(1, len(filtering_particles)),
        reverse=True
    )

    # Append the final state to the return vector.
    smoothing_particles = jnp.concatenate((smoothing_particles, filtering_particles[-1][jnp.newaxis]))

    return smoothing_particles

