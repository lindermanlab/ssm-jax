from typing_extensions import ParamSpec
import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from functools import partial
from jax import vmap
from jax.tree_util import tree_map

from ssm.distributions.expfam import ConjugatePrior, ExponentialFamilyDistribution
from ssm.utils import one_hot


__all__ = ["Bernoulli",
           "IndependentBernoulli",
           "Beta",
           "Categorical",
           "Dirichlet",
           "Gamma",
           "MultivariateNormalFullCovariance",
           "MultivariateNormalTriL",
           "Poisson",
           "IndependentPoisson"
           ]


class Bernoulli(ExponentialFamilyDistribution, tfd.Bernoulli):
    @classmethod
    def from_params(cls, params):
        # it's important to be consistent with how we init classes
        # to avoid re-jit (i.e. logits then probs causes recompile)
        return cls(logits=np.log(params))

    @staticmethod
    def sufficient_statistics(datapoint):
        return datapoint, 1 - datapoint


class IndependentBernoulli(ExponentialFamilyDistribution, tfd.Independent):
    def __init__(self, logits=None, probs=None) -> None:
        parameters = dict(locals())
        super(IndependentBernoulli, self).__init__(
            tfd.Bernoulli(logits=logits, probs=probs), reinterpreted_batch_ndims=1)

        # Ensure that the subclass (not base class) parameters are stored.
        self._parameters = parameters

    def _parameter_properties(self, dtype=np.float32, num_classes=None):
        return dict(
            probs=tfp.internal.parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=tfb.Sigmoid,
                is_preferred=False,
                event_ndims=1),
            logits=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1))

    @property
    def probs(self):
        return self._parameters["probs"]

    @property
    def logits(self):
        return self._parameters["logits"]

    @classmethod
    def from_params(cls, params):
        return cls(probs=params)

    @staticmethod
    def sufficient_statistics(datapoint):
        return datapoint, 1 - datapoint


class Beta(ConjugatePrior, tfd.Beta):

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        concentration1, concentration0 = natural_parameters
        return cls(concentration1, concentration0)

    @property
    def natural_parameters(self):
        return (self.concentration1, self.concentration0)


class Categorical(ExponentialFamilyDistribution, tfd.Categorical):
    @classmethod
    def from_params(cls, params):
        # it's important to be consistent with how we init classes
        # to avoid re-jit (i.e. logits then probs causes recompile)
        return cls(logits=np.log(params))

    @staticmethod
    def sufficient_statistics(datapoint, num_classes):
        raise NotImplementedError("We need to figure out how to plumb number of classes")
        return one_hot(datapoint, num_classes)


class Dirichlet(ConjugatePrior, tfd.Dirichlet):

    @classmethod
    def from_natural_parameters(cls, concentration):
        return cls(concentration)

    @property
    def natural_parameters(self):
        return self.concentration


class Gamma(ConjugatePrior, tfd.Gamma):
    @classmethod
    def from_natural_parameters(cls, params):
        concentration, rate = params
        return cls(concentration, rate)

    @property
    def natural_parameters(self):
        return (self.concentration, self.rate)


class MultivariateNormalFullCovariance(ExponentialFamilyDistribution,
                                       tfd.MultivariateNormalFullCovariance):
    """The multivariate normal distribution is both an exponential family
    distribution as well as a conjugate prior (for the mean of a multivariate
    normal distribution)."""

    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(*params, **kwargs)

    @classmethod
    def from_sufficient_statistics(cls, statistics, **kwargs):
        _, Ex, ExxT, _ = statistics
        cov = ExxT - np.einsum('...d, ...e -> ...de', Ex, Ex)
        return cls(Ex, cov, **kwargs)

    @staticmethod
    def sufficient_statistics(datapoint):
        return (1.0, datapoint, np.outer(datapoint, datapoint), 1.0)

class MultivariateNormalTriL(ExponentialFamilyDistribution,
                             tfd.MultivariateNormalTriL):
    @classmethod
    def from_params(cls, params):
        loc, covariance = params
        return cls(loc, np.linalg.cholesky(covariance))

    @staticmethod
    def sufficient_statistics(datapoint):
        return (1.0, datapoint, np.outer(datapoint, datapoint), 1.0)


class Poisson(ExponentialFamilyDistribution, tfd.Poisson):
    @classmethod
    def from_params(cls, params):
        return cls(log_rate=np.log(params))

    @staticmethod
    def sufficient_statistics(datapoint):
        return (datapoint, np.ones_like(datapoint))


class IndependentPoisson(ExponentialFamilyDistribution, tfd.Independent):
    def __init__(self, rates=None, log_rates=None) -> None:
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
        return cls(log_rates=np.log(params))

    @staticmethod
    def sufficient_statistics(datapoint):
        return (datapoint, np.ones_like(datapoint))
