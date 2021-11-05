import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.distributions.expfam import ConjugatePrior, ExponentialFamilyDistribution
from ssm.utils import one_hot


__all__ = ["Categorical",
           "Dirichlet",
           "Gamma",
           "MultivariateNormalFullCovariance",
           "MultivariateNormalTriL"
           ]


class Categorical(ExponentialFamilyDistribution, tfd.Categorical):
    @classmethod
    def from_params(cls, params):
        return cls(probs=params)

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
