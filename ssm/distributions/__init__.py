from ssm.distributions.expfam import register_prior, ExponentialFamilyDistribution, ConjugatePrior
from ssm.distributions.niw import NormalInverseWishart
from ssm.distributions.mniw import MatrixNormalInverseWishart
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.distributions.linreg import GaussianLinearRegression, GaussianLinearRegressionPrior
from ssm.distributions.wrappers import *


# Register conjugate priors
register_prior(Bernoulli, Beta)
register_prior(IndependentBernoulli, Beta)
register_prior(Binomial, Beta)
register_prior(IndependentBinomial, Beta)
register_prior(Categorical, Dirichlet)
register_prior(GaussianLinearRegression, GaussianLinearRegressionPrior)
register_prior(MultivariateNormalFullCovariance, NormalInverseWishart)
register_prior(MultivariateNormalTriL, NormalInverseWishart)
register_prior(Poisson, Gamma)
register_prior(IndependentPoisson, Gamma)