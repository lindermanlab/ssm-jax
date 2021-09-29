
import jax.numpy as np

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization
# from tensorflow_probability.python.internal import parameter_properties
# from tensorflow_probability.python.internal import prefer_static as ps
# from tfp.bijectors import fill_scale_tril as fill_scale_tril_bijector
# from tensorflow_probability.python.internal import dtype_util

class GaussianLinearRegression(tfp.distributions.Distribution):
    """Linear regression with Gaussian noise.
    """
    def __init__(
        self,
        weights,
        bias,
        scale_tril,
        validate_args=False,
        allow_nan_stats=True,
        name="GaussianLinearRegression",
    ):
        self._weights = weights
        self._bias = bias
        self._scale_tril = scale_tril

        super(GaussianLinearRegression, self).__init__(
            dtype=weights.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(locals()),
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
        ))

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

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def scale(self):
        return np.einsum('...ij,...ji->...ij', self.scale_tril, self.scale_tril)

    def predict(self, covariates=None):
        return tfp.distributions.MultivariateNormalTriL(
            covariates @ self.weights.T + self.bias, self.scale_tril)

    def _sample(self, covariates=None, seed=None, sample_shape=()):
        d = self.predict(covariates)
        # d = tfp.distributions.MultivariateNormalTriL(
            # self.predict(covariates), self.scale_tril)
        return d.sample(sample_shape=sample_shape, seed=seed)

    def _log_prob(self, data, covariates=None, **kwargs):
        d = self.predict(covariates)
        # d = tfp.distributions.MultivariateNormalTriL(
            # self.predict(covariates), self.scale_tril)
        return d.log_prob(data)
