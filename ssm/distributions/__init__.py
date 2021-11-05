from ssm.distributions.expfam import register_prior
from ssm.distributions.niw import NormalInverseWishart
from ssm.distributions.mniw import MatrixNormalInverseWishart
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.distributions.poisson import IndependentPoisson
from ssm.distributions.wrappers import *


# Register conjugate priors
register_prior(MultivariateNormalFullCovariance, NormalInverseWishart)
register_prior(MultivariateNormalTriL, NormalInverseWishart)
register_prior(IndependentPoisson, Gamma)
register_prior(Categorical, Dirichlet)