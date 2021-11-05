import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from ssm.distributions.expfam import ExponentialFamilyDistribution

class IndependentPoisson(ExponentialFamilyDistribution, tfd.Independent):
    def __init__(self, rates, log_rates=None) -> None:
        parameters = dict(locals())
        super(IndependentPoisson, self).__init__(
            tfd.Poisson(rate=rates, log_rate=log_rates), reinterpreted_batch_ndims=1)

        # Ensure that the subclass (not base class) parameters are stored.
        self._parameters = parameters

    def _parameter_properties(self, dtype=np.float32, num_classes=None):
        return dict(
            rates=tfp.internal.parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: tfb.Softplus(
                        low=tfp.internal.dtype_util.eps(dtype))),
                is_preferred=False,
                event_ndims=1),
            log_rates=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1))

    @property
    def rates(self):
        return self._parameters["rates"]

    @property
    def log_rates(self):
        return self._parameters["log_rates"]

    @classmethod
    def from_params(cls, params):
        return cls(rates=params)

    @staticmethod
    def sufficient_statistics(datapoint):
        return (datapoint, np.ones_like(datapoint))
