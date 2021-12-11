"""
SMC filtering/smoothing for SSMs.
"""
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.scipy import special as spsp
from jax import vmap
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization
from typing import Iterable

# SSM imports.
from ssm.utils import Verbosity

# Set the default verbosity.
default_verbosity = Verbosity.DEBUG


def smc(key,
        model,
        dataset,
        initialization_distribution=None,
        proposal=None,
        num_particles=50,
        resampling_criterion=None,
        resampling_function=None,
        use_stop_gradient_resampling=False,
        use_resampling_gradients=False,
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
            Function takes arguments of (dataset, model, particles, time, p_dist, , q_state ...), but doesn't have to
            make use of them.  Allows more expressive initial proposals to be constructed.  The proposal must be a
            function that can be called as fn(...) and returns a TFP distribution object.  Therefore, the proposal
            function may need to be  defined by closing over a proposal class as appropriate.

        proposal (function, No size, default=model.dynamics_distribution):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, model, particles, time, p_dist, , q_state ...), but doesn't have to
            make use of them.

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
            NAND with use_resampling_gradients.

        use_resampling_gradients (bool, No size, default=False):
            Whether to use resampling gradients [Maddison et al, 2017].
            NAND with use_stop_gradient_resampling.

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        particles (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Particles approximating the smoothing distribution.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        z_hat (np.array -- or -- np.float, (batch, ) -- or -- No size):
            Log-normalizer estimated via SMC.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        ancestry (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Full matrix of resampled ancestor indices.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.

        filtering_particles. (np.array, (batch x time x num_particles x state_dim) -- or -- (time x num_particles x state_dim)):
            Particles approximating the filtering distribution.  May have a leading batch dimension if `dataset` also
            has a leading batch dimension.
    """

    # Check the args.
    # Implement NAND using arithmetic.
    assert use_stop_gradient_resampling + use_resampling_gradients != 2, \
        "[Error]: Cannot use both resampling gradients and stop gradient resampling."

    # Assign defaults.
    if resampling_criterion is None:
        resampling_criterion = always_resample_criterion
    if resampling_function is None:
        resampling_function = systematic_resampling

    # Close over the static arguments.
    single_smc_closed = lambda _k, _d: \
        _single_smc(_k, model, _d, initialization_distribution, proposal, num_particles,
                    resampling_criterion, resampling_function, use_stop_gradient_resampling,
                    use_resampling_gradients, verbosity=verbosity)

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
                use_resampling_gradients=False,
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
            Function takes arguments of (dataset, model, particles, time, p_dist, , q_state, ...), but doesn't have to
            make use of them.  Allows more expressive initial proposals to be constructed.  The proposal must be a
            function that can be called as fn(...) and returns a TFP distribution object.  Therefore, the proposal
            function may need to be  defined by closing over a proposal class as appropriate.

        proposal (function, No size, default=model.dynamics_distribution):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, model, particles, time, p_dist, q_state, ...), but doesn't
            have to make
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

        use_resampling_gradients (bool, No size, default=False):
            Whether to use resampling gradients [Maddison et al, 2017].
            NAND with use_stop_gradient_resampling.

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        particles (np.array, (time x num_particles x state_dim)):
            Particles approximating the smoothing distribution.

        z_hat (np.float, No size):
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
    # This function also returns the initial state of the proposal, and so there is no
    # initial proposal state if we are using p.
    if initialization_distribution is None:
        initialization_distribution = lambda *args: (model.initial_distribution(), None)

    # If no explicit proposal is provided, default to BPF.
    # The default BPF implementation uses just the current particles as inputs.
    # More complex proposals allow multiple arguments to be input and are
    # filtered on the inside.
    # This function also returns the iterated state of the proposal, and so there is no
    # initial proposal state if we are using p.
    if proposal is None:
        proposal = lambda *args: (model.dynamics_distribution(args[2]), None)

    # Do the forward pass.
    filtering_particles, z_hat, ancestry, resampled, accumulated_log_incr_weights = \
        _smc_forward_pass(key,
                          model,
                          dataset,
                          initialization_distribution,
                          proposal,
                          num_particles,
                          resampling_criterion,
                          resampling_function,
                          use_stop_gradient_resampling,
                          use_resampling_gradients,
                          verbosity)

    # Now do the backwards pass to generate the smoothing distribution.
    smoothing_particles = _smc_backward_pass(filtering_particles, ancestry, verbosity)

    # Now inscribe the results into an SMCPosterior object for modularity.
    smc_posterior = SMCPosterior.from_params(smoothing_particles,
                                             accumulated_log_incr_weights,
                                             ancestry,
                                             filtering_particles,
                                             z_hat,
                                             resampled)

    return smc_posterior


def _smc_forward_pass(key,
                      model,
                      dataset,
                      initialization_distribution,
                      proposal,
                      num_particles,
                      resampling_criterion,
                      resampling_function,
                      use_stop_gradient_resampling=False,
                      use_resampling_gradients=False,
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
            Function takes arguments of (dataset, model, particles, time, p_dist, q_state, ...), but doesn't have to
            make use of them.  Allows more expressive initial proposals to be constructed.  The proposal must be a
            function that can be called as fn(...) and returns a TFP distribution object.  Therefore, the proposal
            function may need to be  defined by closing over a proposal class as appropriate.

        proposal (function, No size):
            Allows a custom proposal to be used to propose transitions from.  Using default value of None means that
            the prior is used as the proposal.
            Function takes arguments of (dataset, model, particles, time, p_dist, q_state, ...), but doesn't have to
            make use of them.  Allows more expressive proposals to be constructed.  The proposal must be a function that
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
            NAND with use_resampling_gradients.

        use_resampling_gradients (bool, No size, default=False):
            Whether to use resampling gradients [Maddison et al, 2017].
            NAND with use_stop_gradient_resampling.

        verbosity (SSM.verbosity, No size, default=default_verbosity):
            Level of text output.


    :return: Tuple:
        filtering_particles (np.array, (time x num_particles x state_dim)):
            Particles approximating the filtering distribution.

        z_hat (np.float, No size):
            Log-normalizer estimated via SMC.

        ancestors (np.array, (time x num_particles x state_dim)):
            Full matrix of resampled ancestor indices.
    """

    # Generate the initial distribution.
    initial_distribution, initial_q_state = initialization_distribution(dataset, model)

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
        key, particles, accumulated_log_weights, q_state = carry
        key, subkey1, subkey2 = jr.split(key, num=3)

        # Compute the p and q distributions.
        p_dist = model.dynamics_distribution(particles)
        q_dist, q_state = proposal(dataset, model, particles, t, p_dist, q_state)

        # Sample the new particles.
        new_particles = q_dist.sample(seed=subkey1)

        # Compute the incremental importance weight.
        p_log_probability = p_dist.log_prob(new_particles)
        q_log_probability = q_dist.log_prob(new_particles)

        # Test if the observations are NaNs.  If they are NaNs, assign a log-likelihood of zero.
        y_log_probability = jax.lax.cond(np.any(np.isnan(dataset[t])),
                                         lambda _: np.zeros((num_particles, )),
                                         lambda _: model.emissions_distribution(new_particles).log_prob(dataset[t]),
                                         None)
        incremental_log_weights = p_log_probability - q_log_probability + y_log_probability

        # Update the log weights.
        accumulated_log_weights += incremental_log_weights

        # Close over the resampling function.
        closed_do_resample = lambda _crit: do_resample(subkey2, accumulated_log_weights, new_particles, _crit,
                                                       resampling_function,
                                                       use_stop_gradient_resampling=use_stop_gradient_resampling)

        # Resample particles depending on the resampling function chosen.
        # We don't want to resample on the final timestep, so dont...
        resampled_particles, ancestors, resampled_log_weights, should_resample = \
            jax.lax.cond(t == (len(dataset) - 1),
                         lambda *args: closed_do_resample(never_resample_criterion),
                         lambda *args: closed_do_resample(resampling_criterion),
                         None)

        return ((key, resampled_particles, resampled_log_weights, q_state),
                (resampled_particles, accumulated_log_weights, should_resample, ancestors, q_state))

    # Scan over the dataset.
    _, (filtering_particles, log_weights, resampled, ancestors, q_states) = jax.lax.scan(
        smc_step,
        (key, initial_resampled_particles, accumulated_log_weights, initial_q_state),
        np.arange(1, len(dataset)))

    # Need to prepend the initial timestep.
    filtering_particles = np.concatenate((initial_resampled_particles[None, :], filtering_particles))
    log_weights = np.concatenate((initial_incremental_log_weights[None, :], log_weights))
    resampled = np.concatenate((np.asarray([initial_resampled]), resampled))
    ancestors = np.concatenate((np.arange(num_particles)[None, :], ancestors))
    q_states = np.concatenate((np.asarray([initial_q_state]), q_states)) if q_states is not None else None

    # Average over particle dimension.
    p_hats = spsp.logsumexp(log_weights, axis=1) - np.log(num_particles)

    # Compute the log marginal likelihood.
    # Note that this needs to force the last accumulated incremental weight to be used.
    l_tilde = np.sum(p_hats[:-1] * resampled[:-1]) + p_hats[-1]

    # If we are using the resampling gradients, modify the marginal likelihood to contain that gradient information.
    if use_resampling_gradients:
        raise NotImplementedError("This code isn't ready yet.  There are still too many issues with indexing etc.")
        # z_hat = z_hat + _compute_resampling_grad()
        
    return filtering_particles, l_tilde, ancestors, resampled, log_weights


def _smc_backward_pass(filtering_particles, ancestors, verbosity=default_verbosity):
    r"""Do the backwards pass (pruning) given a SINGLE forward pass.

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


def never_resample_criterion(unused_log_weights, unused_t):
    r"""A criterion that never resamples."""
    return False


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
    # NOTE!  Finally found the problem, digitize treats everything to the left of the zeroth bin index as the zeroth
    # bin (which is kind of wild) so we have to subtract one to make sure the indexing is correct as we never show it
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

    if use_stop_gradient_resampling:
        resamp_log_ws = jax.lax.cond(should_resample,
                                     lambda _: log_weights[ancestors] - jax.lax.stop_gradient(log_weights[ancestors]),
                                     lambda _: log_weights,
                                     None)
    else:
        resamp_log_ws = (1. - should_resample) * log_weights

    return resampled_particles, ancestors, resamp_log_ws, should_resample


class SMCPosterior(tfd.Distribution):
    """
    Define a thin wrapper and constructor for a MixtureSameFamily distribution for constructing a representation
    of the SMC smoothing distribution.  Note that the rest of the SMC code considers (T x N x D), but here we think
    about things as (N x T x D), since this is essentially a mixture distribution over the N particles.
    """

    def __init__(self,
                 smoothing_particles,
                 log_weights,
                 ancestry,
                 filtering_particles,
                 log_marginal_likelihood,
                 resampled,
                 has_state_dim=True,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="SMCPosterior", ):
        """
        See from_params.
        """

        self._smoothing_particles = smoothing_particles
        self._log_accumulated_weights = log_weights
        self._ancestry = ancestry
        self._filtering_particles = filtering_particles
        self._log_marginal_likelihood = log_marginal_likelihood
        self._resampled = resampled
        self._has_state_dim = has_state_dim

        # We would detect the dtype dynamically but that would break vmap
        # see https://github.com/tensorflow/probability/issues/1271
        dtype = np.float32
        super(SMCPosterior, self).__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(smoothing_particles=self._smoothing_particles,
                            log_weights=self._log_accumulated_weights,
                            ancestry=self._ancestry,
                            filtering_particles=self._filtering_particles,
                            log_marginal_likelihood=self._log_marginal_likelihood,
                            resampled=self._resampled,
                            has_state_dim=has_state_dim),
            name=name,
        )

    @classmethod
    def from_params(cls,
                    smoothing_particles,
                    log_weights,
                    ancestry,
                    filtering_particles,
                    log_marginal_likelihood,
                    resampled, ):
        """

        Preprocess and instantiate a SMCPosterior object.

        The SMC sweep often works with objects that are (time x particles x state), whereas the underlying distribution
        objects need items to be of shape (particles x time x state).  This re-ordering causes chaos in the __init__
        when jitting, so always instantiate through this function.  Also provides a place for some validation of
        arguments etc.

        This can also be constructed as a PyTree, and hence can have arbitrary leading batch dimensions if multiple
        repeat sweeps / trials / batches have been run.

        # TODO - this currently required that smoothing_particles is a tensor.  Needs to be converted to allow PyTrees.

        Args:

            smoothing_particles (ndarray, (time x num_particles x state_dim)):
                ndarray of the smoothing particles.  Will form the atomic representations used in the mixture.

            log_weights (ndarray, (time x num_particles)):

            ancestry (ndarray, (time x num_particles x state_dim)):
                ndarray of the ancestors of each particle.  Can be omitted for computational savings.

            filtering_particles (ndarray, (time x num_particles x state_dim)):
                ndarray of the filtering particles.  Can be omitted for computational savings.

            log_marginal_likelihood (float):
                The estimated log marginal likelihood.

            resampled (ndarray, (time, )):
                Whether particles were resampled at that time bin.

        Returns: Object of type SMCPosterior.

        """

        # # Validate the args.
        # assert smoothing_particles.shape[0] == log_weights.shape[0], \
        #     "Must be the same number of timesteps."
        # assert smoothing_particles.shape[1] == log_weights.shape[1], \
        #     "Must be the same number of weights as particles."
        #
        # assert smoothing_particles.shape[0] == ancestry.shape[0], \
        #     "[Error]: Ancestry length must match particles length."
        # assert smoothing_particles.shape[1] == ancestry.shape[1], \
        #     "[Error]: Ancestry dim must match n particles."
        #
        # assert filtering_particles.shape == smoothing_particles.shape, \
        #     "[Error]: Filtering particles and smoothing particles must have same shape."

        # If the weights and the particles are the same shape, then there is no state dimension.
        has_state_dim = log_weights.shape != smoothing_particles.shape
        if has_state_dim:
            smoothing_particles = np.moveaxis(smoothing_particles, -3, -2)
            filtering_particles = np.moveaxis(filtering_particles, -3, -2)
        else:
            smoothing_particles = np.moveaxis(smoothing_particles, -2, -1)
            filtering_particles = np.moveaxis(filtering_particles, -2, -1)

        log_weights = np.moveaxis(log_weights, -2, -1)
        ancestry = np.moveaxis(ancestry, -2, -1)

        return cls(smoothing_particles,
                   log_weights,
                   ancestry,
                   filtering_particles,
                   log_marginal_likelihood,
                   resampled,
                   has_state_dim)

    def __len__(self):
        raise NotImplementedError("Len on this object is so ambiguous we disable it.")

    def __getitem__(self, item):
        assert len(self.log_normalizer.shape) > 0, "Cannot directly index inside single posterior."
        return SMCPosterior(**jax.tree_map(lambda args: args[item] if isinstance(args, Iterable) else args, self._parameters))

    def tree_flatten(self):
        children = (self._smoothing_particles,
                    self._log_accumulated_weights,
                    self._ancestry,
                    self._filtering_particles,
                    self._log_marginal_likelihood,
                    self._resampled, )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def _parameter_properties(self, dtype, num_classes=None):
        """
        There is something weird in how i'm constructing the class, but it doesn't seem to make too much of a
        difference.  Without overriding the class property, TFP throws a rather unhelpful warning.

        Need to double check that the dictionary that is returned here is correct.

        see https://github.com/tensorflow/probability/issues/1458

        :param dtype:
        :param num_classes:
        :return:
        """
        return dict(
            smoothing_particles=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            ancestry=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            filtering_particles=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_marginal_likelihood=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            resampled=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
        )

    def _event_shape_tensor(self):
        return tf.TensorShape(self._smoothing_particles.shape[1:])

    def _event_shape(self):
        return self._event_shape_tensor()

    def _batch_shape(self):
        return tf.TensorShape(self._smoothing_particles.shape[0])

    @property
    def state_dimension(self):
        return self._smoothing_particles.shape[-1]

    @property
    def num_timesteps(self):
        return self._smoothing_particles.shape[-2]

    @property
    def num_particles(self):
        return self._smoothing_particles.shape[-3]

    @property
    def log_normalizer(self):
        return self._log_marginal_likelihood

    def _gen_dist(self):
        """
        TODO - This function generates the smoothing distribution as a TFP object.  This is kind clunky though.
        It is because when you call __init__, if you generate this distribution statically ahead of time, when
        `vmap`ing you just get `object` errors... need to use the `.new` method to try and get around this.

        Returns:

        """
        smc_mixture_dist = tfd.Categorical(logits=self.final_particle_weights_unnormalized)   # Weighting distribution.
        particle_dist = tfd.Deterministic(self._smoothing_particles)  # Convert into a set of deterministic dists.

        # Construct the components.  If there are no state dims then there are only one dim.
        if self._has_state_dim:
            smc_components_dist = tfd.Independent(particle_dist, reinterpreted_batch_ndims=2)
        else:
            smc_components_dist = tfd.Independent(particle_dist, reinterpreted_batch_ndims=1)

        return tfd.MixtureSameFamily(smc_mixture_dist, smc_components_dist)

    def _log_prob(self, data, **kwargs):
        """
        IMPORTANT: This is a wild function.  When there is just a single SMC sweep (and particles are the batch
        dimension), this function will return the log prob of a particle as a number under that single sweep.

        If there are multiple sweeps (the batch dimension is repeats or trials) then this function will return a
        tensor of logprobs, equivilant to vmapping over any of the leading dimensions.  This is the most general of
        operation modes, but it does mean that there may need to be some post-hoc shape checking and validation.

        Args:
            data:
            **kwargs:

        Returns:

        """
        return self._gen_dist().log_prob(data)

    def _sample_n(self, n, seed=None, **kwargs):
        """
        Generate the distribution object and then route this call through to the underlying distribution.

        Args:
            n (int):            Number of i.i.d. samples to draw.
            seed (JaxPRNGKey):  Seed the sample.
            **kwargs:           Not used here.

        Returns (ndarray, shape=(n x self._event_shape)):  Sampled values.

        """
        return self._gen_dist()._sample_n(n, seed=seed)

    @property
    def weighted_particles(self):
        return self._smoothing_particles, self.final_particle_weights_unnormalized

    @property
    def final_particle_weights_unnormalized(self):
        """
        Return the final unnormalized accumulated importance weights of the particles.  These are the logits that should
        be used for resampling smoothing particles.

        Note that since these weights are unnormalized, they only make sense as logits when normalized across a batch.

        Returns:

        """
        return self._log_accumulated_weights[..., :, -1]

    @property
    def resampled(self):
        return self._resampled

    @property
    def filtering_particles(self):
        return self._filtering_particles

    @property
    def accumulated_incremental_importance_weights(self):
        return self._weights

    @property
    def incremental_importance_weights(self):
        """
        We need to implement this such that we can get the raw \alpha values out.  This requires parsing the
        resampled vector to see if they have been forcibly zeroed.
        Returns:

        """
        raise NotImplementedError()

    def _mean(self):
        """
        This property doesn't really make sense here.
        :return:
        """
        raise NotImplementedError()

    @property
    def expected_states(self):
        """
        This property doesn't really make sense here.
        :return:
        """
        # return self.mean()
        raise NotImplementedError()

    @property
    def is_stationary(self):
        """
        This property doesn't really make sense here.
        :return:
        """
        raise NotImplementedError()


def _plot_single_sweep(particles, true_states, tag='', preprocessed=False, fig=None, _obs=None):
    """
    Some stock code for plotting the results of an SMC sweep.
    :param particles:
    :param true_states:
    :param tag:
    :param preprocessed:
    :param fig:
    :return:
    """
    # Define the standard plotting colours.
    color_names = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple"
    ]

    gen_label = lambda _k, _s: _s if _k == 0 else None

    if not preprocessed:
        single_sweep_median = np.median(particles, axis=0)
        single_sweep_lsd = np.quantile(particles, 0.17, axis=0)
        single_sweep_usd = np.quantile(particles, 0.83, axis=0)
    else:
        single_sweep_median = particles[0]
        single_sweep_lsd = particles[1]
        single_sweep_usd = particles[2]

    ts = np.arange(len(true_states))

    if fig is not None:
        plt.figure(fig.number)
        plt.clf()
    else:
        fig = plt.figure(figsize=(14, 6))

    for _i, _c in zip(range(single_sweep_median.shape[1]), color_names):
        plt.plot(ts, single_sweep_median[:, _i], c=_c, label=gen_label(_i, 'Predicted'))
        plt.fill_between(ts, single_sweep_lsd[:, _i], single_sweep_usd[:, _i], color=_c, alpha=0.1)

        plt.plot(ts, true_states[:, _i], c=_c, linestyle='--', label=gen_label(_i, 'True'))

    # Enable plotting obs here.
    # if _obs is not None:
    #     for _i, _c in zip(range(_obs.shape[1]), color_names):
    #         plt.scatter(ts, _obs[:, _i], c=_c, marker='.', label=gen_label(_i, 'Observed'))

    plt.title(tag)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    # plt.xlim(-0.5, 3.5)
    plt.pause(0.1)

    return fig


# def _compute_resampling_grad(log_weights, z_hat, ancestors, raoblackwellize_estimator=True):
#     """
#
#     :param log_weights:
#     :param z_hat:
#     :param ancestors:
#     :param raoblackwellize_estimator:
#     :return:
#     """
#
#
#     # TO-DO - check if Scibior paper can use adaptive resampling.
#     # Check that we were always resampling.
#     assert np.all(resampling_function == always_resample_criterion), \
#         "Error: Currently enforcing that always resampling if using Resampling gradients.  We need to derive " \
#         "whether the resampling gradient estimator is correct with variable resampling schedule. "
#
#     # Set whether we are raoblackwellizing the estimator.  This is an internal check and should always be set to
#     # true once we have verified that it is correct.
#     raoblackwellize_estimator = True
#
#     if raoblackwellize_estimator:
#         assert np.all(resampling_function == always_resample_criterion), \
#             "Error: Currently enforcing that always resampling if R-B-ing.  We need to derive " \
#             "precisely how to R-B with variable resampling schedule. "
#
#     # Pull out dimensions.
#     t, n, _ = log_weights.shape
#
#     # Average log_weight at each timestep.
#     p_hats = spsp.logsumexp(log_weights, axis=1) - np.log(n)
#
#     # (T, num_particles)
#     log_weights_normalized = log_weights - spsp.logsumexp(log_weights, axis=1, keepdims=True)
#
#     # If we are R-B-ing, then we need to modify the p-hat terms in the score function.
#     # See (8) in VSMC.
#     if raoblackwellize_estimator:
#         # ``p_hat_diffs[t] = log [p-hat(y_{1:T}) / p-hat(y_{1:t})]`` with length T - 1 (cf. (8) in VSMC)
#         # TO-DO - this cumsum only runs until the penultimate step.
#         p_hat_diffs = jax.lax.stop_gradient(z_hat - np.cumsum(p_hats)[:-1])  # last entry not used
#     else:
#         p_hat_diffs = jax.lax.stop_gradient(z_hat)
#
#     resampling_loss = np.sum()
#
#     # # TO-DO - NOTE - AW - I THINK THIS SHOULD BE TO TIME T.
#     # # TO-DO - decide if having initial ancestors changes everything.
#     # # The anscestor indices may be one out or they may not be.
#     # # If not rao-blackwellized then this should def got to T, if it is R-B, then maybe the last term should be zero
#     # # or maybe the last term can be ignored as it is zero.  Write a little debug for this.
#     # resampling_loss = 0.0
#     # for _t in range(t - 1):
#     #     # Again cf. (8) in VSMC paper
#     #     # (TO-DO: should steps where resampling doesn't happen be zeroed out? For now just doing `always_resample`)
#     #     # resampling_loss += p_hat_diffs[t] * log_weights_normalized[t, ancestors[t]].sum()
#     #
#     #     # resampling_loss += jax.lax.stop_gradient(resampled[t]) * \
#     #                          p_hat_diffs[t] * log_weights_normalized[t, ancestors[t]].sum()
#     #
#     #     # Non-Rao Blackwellized gradient
#     #     resampling_loss += jax.lax.stop_gradient(z_hat) * \
#     #                        log_weights_normalized[t, ancestors[t]].sum()
#
#     # Add a numerical zero to the marginal likelihood, but stop the gradient through one of the terms so that the
#     # gradient of the resampling loss is added to the gradient of the vanilla log marginal term.
#     return resampling_loss - jax.lax.stop_gradient(resampling_loss)
