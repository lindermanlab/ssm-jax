
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
                 validate_args=False,
                 allow_nan_stats=True,
                 name="SMCPosterior", ):
        """

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

        # Note that these require the batch dimension (T) to be the first dimension.
        self._ancestry = ancestry
        self._smoothing_particles = smoothing_particles
        self._filtering_particles = filtering_particles
        self._log_accumulated_weights = log_weights
        self._log_marginal_likelihood = log_marginal_likelihood
        self._resampled = resampled

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
                            resampled=self._resampled,),
            name=name,
        )


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
        # If there are only three dimensions, then the particle dimension is the batch dimension.
        if len(self._smoothing_particles.shape) == 3:
            return tf.TensorShape((self._smoothing_particles.shape[0],
                                   self._smoothing_particles.shape[2]))
        else:
            # Otherwise, the leading dimension in the batch dim.
            return tf.TensorShape(self._smoothing_particles.shape[1:])

    def _event_shape(self):
        return self._event_shape_tensor()

    def _batch_shape(self):
        return tf.TensorShape(self._smoothing_particles.shape[0])

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

    def _gen_dist(self):
        # Construct the weighting distribution.
        smc_mixture_distribution = tfd.Categorical(logits=self.final_particle_weights)

        # Convert the weights into a set of deterministic distributions.
        particle_dist = tfd.Deterministic(np.moveaxis(self.particles, -3, -2))

        # Construct the components.
        smc_components_distribution = tfd.Independent(particle_dist, reinterpreted_batch_ndims=2)

        return tfd.MixtureSameFamily(smc_mixture_distribution, smc_components_distribution)

    # Define the properties and class methods.
    # These definitions predominantly follow the order of HMM -> Posterior.py -> _HMMPosterior .

    def __len__(self):
        raise NotImplementedError("Len on this object is so ambiguous we disable it.")

    @property
    def state_dimension(self):
        return self._smoothing_particles.shape[-1]

    @property
    def num_particles(self):
        return self._smoothing_particles.shape[-2]

    @property
    def num_timesteps(self):
        return self._smoothing_particles.shape[-3]

    @property
    def is_stationary(self):
        """
        This property doesn't really make sense here.
        :return:
        """
        raise NotImplementedError()

    @property
    def log_normalizer(self):
        return self._log_marginal_likelihood

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

    def _log_prob(self, data, **kwargs):
        there is a reasonably substantial bug in here somewhere.
        return self._gen_dist().log_prob(data)

    def _sample_n(self, n, seed=None, **kwargs):
        """
        Generate the distribution object and then route this call through to the underlying distribution.
        :param n:
        :param seed:
        :param kwargs:
        :return:
        """
        return self._gen_dist()._sample_n(n, seed=seed)

    # These properties are then SMCPosterior specific.

    @property
    def particles(self):
        return self._smoothing_particles

    @property
    def incremental_importance_weights(self):
        raise NotImplementedError()
        # return self._

    @property
    def final_particle_weights(self):
        return self._log_accumulated_weights[..., -1, :]

    @property
    def resampled(self):
        return self._resampled

    @property
    def filtering_particles(self):
        return self._filtering_particles  # .

    @property
    def accumulated_incremental_importance_weights(self):
        return self._weights
