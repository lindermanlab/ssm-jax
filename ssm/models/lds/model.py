"""
LDS Model Classes
=================
"""
import jax.numpy as np
from jax.tree_util import register_pytree_node_class, tree_map
from ssm.distributions.glm import BernoulliGLM, GaussianGLM, PoissonGLM
from ssm.distributions.linreg import GaussianLinearRegression
from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.inference.em import em
from ssm.inference.laplace_em import _laplace_e_step, laplace_em
from ssm.models.base import SSM
from tensorflow_probability.substrates import jax as tfp

from ssm.utils import Verbosity

from .dynamics import GaussianLinearRegressionDynamics
from .emissions import (BernoulliGLMEmissions,
                        GaussianLinearRegressionEmissions, PoissonGLMEmissions)
from .initial_distributions import GaussianInitialDistribution

# supported emissions classes for GLM-LDS
_GLM_DISTRIBUTIONS = [
    GaussianGLM,
    PoissonGLM,
    BernoulliGLM
]

@register_pytree_node_class
class LDS(SSM):
    def __init__(self,
                 initial_distribution,
                 dynamics_distribution,
                 emissions_distribution):
        """TODO

        Args:
            initial_distribution ([type]): [description]
            dynamics_distribution ([type]): [description]
            emissions_distribution ([type]): [description]
        """

        # TODO: better parameter checking
        assert isinstance(initial_distribution, tfp.distributions.MultivariateNormalTriL)
        assert isinstance(dynamics_distribution, GaussianGLM) or isinstance(dynamics_distribution, GaussianLinearRegression)
        assert type(emissions_distribution) in _GLM_DISTRIBUTIONS

        # initialize initial_distribution, dynamics, and emissions components
        # TODO: make look-up table
        self.initials = GaussianInitialDistribution(distribution=initial_distribution)
        self.dynamics = GaussianLinearRegressionDynamics(distribution=dynamics_distribution)
        if isinstance(emissions_distribution, GaussianGLM):
            self.emissions = GaussianLinearRegressionEmissions(distribution=emissions_distribution)
            # TODO can we have a constructor that returns appropriate child instance?
        elif isinstance(emissions_distribution, PoissonGLM):
            self.emissions = PoissonGLMEmissions(distribution=emissions_distribution)
        elif isinstance(emissions_distribution, BernoulliGLM):
            self.emissions = BernoulliGLMEmissions(distribution=emissions_distribution)

    @property
    def latent_dim(self):
        return self.emissions.distribution.weights.shape[-1]

    @property
    def emissions_dim(self):
        return self.emissions.distribution.weights.shape[-2]

    @property
    def initial_mean(self):
        return self.initials.distribution.loc

    @property
    def initial_covariance(self):
        return self.initials.distribution.covariance()

    @property
    def dynamics_matrix(self):
        return self.dynamics.distribution.weights

    @property
    def dynamics_bias(self):
        return self.dynamics.distribution.bias

    @property
    def dynamics_noise_covariance(self):
        Q_sqrt = self.dynamics.distribution.scale_tril
        return Q_sqrt @ Q_sqrt.T

    @property
    def emissions_matrix(self):
        return self.emissions.distribution.weights

    @property
    def emissions_bias(self):
        return self.emissions.distribution.bias

    def tree_flatten(self):
        children = (self.initials.distribution,
                    self.dynamics.distribution,
                    self.emissions.distribution)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        initial_distribution, dynamics_distribution, emissions_distribution = children

        return cls(initial_distribution,
                   dynamics_distribution,
                   emissions_distribution)

    def initial_distribution(self):
        return self.initials.distribution

    def dynamics_distribution(self, state):
        return self.dynamics.distribution.predict(state)

    def emissions_distribution(self, state):
        return self.emissions.distribution.predict(state)

    # Inference Routines

    def approximate_posterior(self, data, initial_states=None):
        """Approximate E step
        """
        return _laplace_e_step(self, data, initial_states)

    def m_step(self, data, posterior, prior=None, rng=None):
        """M step for model
        """
        initial_distribution = self.initials.distribution  # TODO initial dist needs prior
        transition_distribution = self.dynamics.exact_m_step(data, posterior, prior=prior)
        emissions_distribution = self.emissions.approx_m_step(data, posterior, rng=rng)  # TODO can this cause 2x jit?
        return LDS(initial_distribution,
                   transition_distribution,
                   emissions_distribution)

    def fit(self, data, method="laplace_em", rng=None, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG, **kwargs):

        # ensure data has a batch dimension
        single_batch_mode = False
        if data.ndim == 2:
            single_batch_mode = True
            data = np.expand_dims(data, axis=0)
        assert data.ndim == 3, "data must have a batch dimension (B, T, N)"

        model = self
        if method == "laplace_em":
            elbos, lds, posteriors = laplace_em(rng, model, data, num_iters=num_iters, tol=tol, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        # squeeze first dimension
        if single_batch_mode:
            posteriors = tree_map(lambda x: x[0], posteriors)

        return elbos, lds, posteriors

@register_pytree_node_class
class GaussianLDS(LDS):
    def __init__(self,
                 initial_distribution,
                 dynamics_distribution,
                 emissions_distribution):
        """TODO

        Args:
            initial_distribution ([type]): [description]
            dynamics_distribution ([type]): [description]
            emissions_distribution ([type]): [description]
        """
        assert isinstance(initial_distribution, tfp.distributions.MultivariateNormalTriL)
        assert isinstance(dynamics_distribution, GaussianLinearRegression)
        assert isinstance(emissions_distribution, GaussianLinearRegression)
        self.initials = GaussianInitialDistribution(distribution=initial_distribution)
        self.dynamics = GaussianLinearRegressionDynamics(distribution=dynamics_distribution)
        self.emissions = GaussianLinearRegressionEmissions(distribution=emissions_distribution)

    @property
    def emissions_noise_covariance(self):
        R_sqrt = self.emissions.distribution.scale_tril
        return R_sqrt @ R_sqrt.T

    def natural_parameters(self, data):
        """ TODO
        """
        seq_len = data.shape[0]

        # Shorthand names for parameters
        m1 = self.initial_mean
        Q1 = self.initial_covariance
        A = self.dynamics_matrix
        b = self.dynamics_bias
        Q = self.dynamics_noise_covariance
        C = self.emissions_matrix
        d = self.emissions_bias
        R = self.emissions_noise_covariance

        # diagonal blocks of precision matrix
        J_diag = np.dot(C.T, np.linalg.solve(R, C))  # from observations
        J_diag = np.tile(J_diag[None, :, :], (seq_len, 1, 1))
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag = J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag = J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        h = np.dot(data - d, np.linalg.solve(R, C))  # from observations
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h = h.at[1:].add(np.linalg.solve(Q, b))

        # should J_obs, J_dyn, J_init be separate?
        return J_diag, J_lower_diag, h

    # Methods for inference
    def e_step(self, data):
        return MultivariateNormalBlockTridiag(*self.natural_parameters(data))

    def m_step(self, data, posterior, prior=None, rng=None):
        initial_distribution = self.initials.distribution  # TODO: initial needs prior
        transition_distribution = self.dynamics.exact_m_step(data, posterior, prior=prior)
        emissions_distribution = self.emissions.exact_m_step(data, posterior, prior=prior)
        return GaussianLDS(initial_distribution, transition_distribution, emissions_distribution)

    def marginal_likelihood(self, data, posterior=None):
        """The exact marginal likelihood of the observed data.

            For a Gaussian LDS, we can compute the exact marginal likelihood of
            the data (y) given the posterior p(x | y) via Bayes' rule:

            .. math::
                \log p(y) = \log p(y, x) - \log p(x | y)

            This equality holds for _any_ choice of x. We'll use the posterior mean.

            Args:
                - lds (LDS): The LDS model.
                - data (array, (num_timesteps, obs_dim)): The observed data.
                - posterior (MultivariateNormalBlockTridiag):
                    The posterior distribution on the latent states. If None,
                    the posterior is computed from the `lds` via message passing.
                    Defaults to None.

            Returns:
                - lp (float): The marginal log likelihood of the data.
            """
        if posterior is None:
            posterior = self.e_step(data)
        states = posterior.mean
        return self.log_probability(states, data) - posterior.log_prob(states)

    def fit(self, data, method="em", rng=None, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):

            single_batch_mode = False
            # ensure data has a batch dimension
            if data.ndim == 2:
                single_batch_mode = True
                data = np.expand_dims(data, axis=0)
            assert data.ndim == 3, "data must have a batch dimension (B, T, N)"

            model = self
            kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

            if method == "em":
                elbos, lds, posteriors = em(model, data, **kwargs)
            elif method == "laplace_em":
                if rng is None:
                    raise ValueError("Laplace EM requires a PRNGKey. Please provide an rng to fit.")
                elbos, lds, posteriors = laplace_em(rng, model, data, **kwargs)
            else:
                raise ValueError(f"Method {method} is not recognized/supported.")

            # squeeze first dimension
            if single_batch_mode:
                posteriors = tree_map(lambda x: x[0], posteriors)

            return elbos, lds, posteriors

