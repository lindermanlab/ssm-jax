
import jax.numpy as np

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization

from ssm.distributions.linreg import GaussianLinearRegression


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

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def _linear_prediction(self, covariates=None):
        return covariates @ self.weights.T + self.bias

    @staticmethod
    def _mean_function(predicted_linear_response):
        """
        Function mapping the predicted linear response to the mean of the noise distribution.

        Inverse of the link function g^{-1}.
        """
        return NotImplementedError

    def _get_noise_distribution(self, mean):
        raise NotImplementedError

    def predict(self, covariates=None):
        return self._get_noise_distribution(
            self._mean_function(self._linear_prediction(covariates=covariates)))

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
        return np.exp(predicted_linear_response)

    def _get_noise_distribution(self, mean):
        return tfp.distributions.Independent(
            tfp.distributions.Poisson(mean),
            reinterpreted_batch_ndims=1
        )


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
        return 1/(1 + np.exp(-1 * predicted_linear_response))

    def _get_noise_distribution(self, mean):
        return tfp.distributions.Independent(
            tfp.distributions.Bernoulli(probs=mean),
            reinterpreted_batch_ndims=1
        ) 


class GaussianGLM(GeneralizedLinearModel):
    def __init__(
        self,
        weights,
        bias,
        scale_tril,
        validate_args=False,
        allow_nan_stats=True,
        name="GaussianGLM",
    ):
        self._scale_tril = scale_tril

        parameters = dict(weights=weights,
                          bias=bias,
                          scale_tril=scale_tril)

        super(GaussianGLM, self).__init__(
            weights=weights,
            bias=bias,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            scale_tril=tfp.internal.parameter_properties.ParameterProperties(
                event_ndims=2,
                # shape_fn=lambda sample_shape: ps.concat([sample_shape, sample_shape[-1:]], axis=0),
                # default_constraining_bijector_fn=lambda: tfp.bijectors.fill_scale_tril.FillScaleTriL(diag_shift=dtype_util.eps(dtype)))
            )
        )

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def scale(self):
        return np.einsum('...ij,...ji->...ij', self.scale_tril, self.scale_tril)

    def _mean_function(self, predicted_linear_response):
        return predicted_linear_response

    def _get_noise_distribution(self, mean):
        return tfp.distributions.MultivariateNormalTriL(mean, self._scale_tril)