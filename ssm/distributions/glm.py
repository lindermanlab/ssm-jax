import jax.numpy as np
from jax.nn import softplus

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization


class GeneralizedLinearModel(tfp.distributions.Distribution):
    """Linear regression with Exponential Family Noise
    """
    def __init__(
        self,
        weights,
        bias,
        validate_args=False,
        allow_nan_stats=True,
        parameters=None,
        name="GeneralizedLinearModel",
    ):

        self._weights = weights
        self._bias = bias

        # default parameter set if not overridden
        if parameters is None:
            parameters = dict(weights=weights, bias=bias)

        super(GeneralizedLinearModel, self).__init__(
            dtype=weights.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1))

    @property
    def data_dimension(self):
        return self.weights.shape[-2]

    @property
    def covariate_dimension(self):
        return self.weights.shape[-1]

    def _event_shape(self):
        return self.weights.shape[-2]

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def _linear_prediction(self, covariates=None):
        return np.einsum('...i,...ji->...j', covariates, self.weights) + self.bias

    @staticmethod
    def _mean_function(predicted_linear_response):
        """
        Function mapping the predicted linear response to the parameters of the noise distribution.

        Generally speaking, the parameters should be the mean, in which case this would be
        the inverse of the link function g^{-1}. However, we may choose to parameterize our
        distribution as, e.g., log mean, so this is a more general function.
        """
        return NotImplementedError

    def _get_noise_distribution(self, params):
        raise NotImplementedError

    def predict(self, covariates=None):
        return self._get_noise_distribution(
            self._mean_function(
                self._linear_prediction(covariates=covariates)))

    def _sample(self, covariates=None, seed=None, sample_shape=()):
        d = self.predict(covariates)
        return d.sample(sample_shape=sample_shape, seed=seed)

    def _log_prob(self, data, covariates=None, **kwargs):
        d = self.predict(covariates)
        return d.log_prob(data)


class PoissonGLM(GeneralizedLinearModel):
    def __init__(
        self,
        weights,
        bias,
        validate_args=False,
        allow_nan_stats=True,
        name="PoissonGLM",
    ):

        super(PoissonGLM, self).__init__(
            weights=weights,
            bias=bias,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # TODO revisit parameter properties at some point
        # pylint: disable=g-long-lambda
        return dict(
            weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1))


    def _mean_function(self, predicted_linear_response):
        # return np.exp(predicted_linear_response)
        return softplus(predicted_linear_response) + 1e-4

    def _get_noise_distribution(self, params):
        return tfp.distributions.Independent(
            tfp.distributions.Poisson(rate=params),
            reinterpreted_batch_ndims=1
        )

    # def predict(self, covariates=None):
    #     # Override the predict function to use the log rate directly
    #     return tfp.distributions.Independent(
    #         tfp.distributions.Poisson(
    #             log_rate=self._linear_prediction(covariates=covariates)),
    #         reinterpreted_batch_ndims=1)


class BernoulliGLM(GeneralizedLinearModel):
    def __init__(
        self,
        weights,
        bias,
        validate_args=False,
        allow_nan_stats=True,
        name="BernoulliGLM",
    ):

        super(BernoulliGLM, self).__init__(
            weights=weights,
            bias=bias,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1))

    def _mean_function(self, predicted_linear_response):
        """Mean function is sigmoid
        """
        return 1/(1 + np.exp(-1 * predicted_linear_response))

    def _get_noise_distribution(self, params):
        return tfp.distributions.Independent(
            tfp.distributions.Bernoulli(probs=params),
            reinterpreted_batch_ndims=1
        )

    def predict(self, covariates=None):
        # Override the predict function to use the logits directly
        return tfp.distributions.Independent(
            tfp.distributions.Bernoulli(
                logits=self._linear_prediction(covariates=covariates)),
            reinterpreted_batch_ndims=1)
