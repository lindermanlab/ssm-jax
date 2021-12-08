
import jax
import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf


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
        return SMCPosterior(**jax.tree_map(lambda args: args[item], self._parameters))

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
