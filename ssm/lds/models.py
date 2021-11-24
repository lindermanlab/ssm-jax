import inspect

import jax.numpy as np
import jax.random as jr
from jax.tree_util import register_pytree_node_class

from ssm.distributions.mvn_block_tridiag import MultivariateNormalBlockTridiag
from ssm.inference.em import em
from ssm.inference.laplace_em import laplace_em
from ssm.lds.base import LDS
from ssm.lds.initial import StandardInitialCondition
from ssm.lds.dynamics import StationaryDynamics
from ssm.lds.emissions import GaussianEmissions, PoissonEmissions
from ssm.utils import Verbosity, format_dataset, random_rotation, make_named_tuple


LDSPosterior = MultivariateNormalBlockTridiag


@register_pytree_node_class
class GaussianLDS(LDS):
    def __init__(self,
                 num_latent_dims: int,
                 num_emission_dims: int=None,
                 initial_state_mean: np.ndarray=None,
                 initial_state_scale_tril: np.ndarray=None,
                 dynamics_weights: np.ndarray=None,
                 dynamics_bias: np.ndarray=None,
                 dynamics_scale_tril: np.ndarray=None,
                 emission_weights: np.ndarray=None,
                 emission_bias: np.ndarray=None,
                 emission_scale_tril: np.ndarray=None,
                 seed: jr.PRNGKey=None):
        """LDS with Gaussian emissions.

        .. math::
            p(y_t | x_t) \sim \mathcal{N}(\mu_{x_t}, \Sigma_{x_t})

        The GaussianLDS can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_latent_dims``, ``num_emission_dims``, and ``seed``
        to create a GaussianLDS with generic, randomly initialized parameters.

        Args:
            num_latent_dims (int): number of latent dims
            num_emission_dims (int, optional): number of emissions dims. Defaults to None.
            initial_state_mean (np.ndarray, optional): [description]. Defaults to None.
            initial_state_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            dynamics_weights (np.ndarray, optional): [description]. Defaults to None.
            dynamics_bias (np.ndarray, optional): [description]. Defaults to None.
            dynamics_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            emission_weights (np.ndarray, optional): [description]. Defaults to None.
            emission_bias (np.ndarray, optional): [description]. Defaults to None.
            emission_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            seed (jr.PRNGKey, optional): [description]. Defaults to None.
        """

        self._parameters = make_named_tuple(dict_in=locals(),
                                            keys=list(inspect.signature(self.__init__)._parameters.keys()),
                                            name=str(self.__class__.__name__) + 'Tuple')

        if initial_state_mean is None:
            initial_state_mean = np.zeros(num_latent_dims)

        if initial_state_scale_tril is None:
            initial_state_scale_tril = np.eye(num_latent_dims)

        if dynamics_weights is None:
            seed, rng = jr.split(seed, 2)
            dynamics_weights = random_rotation(rng, num_latent_dims, theta=np.pi/20)

        if dynamics_bias is None:
            dynamics_bias = np.zeros(num_latent_dims)

        if dynamics_scale_tril is None:
            dynamics_scale_tril = 0.1**2 * np.eye(num_latent_dims)

        if emission_weights is None:
            seed, rng = jr.split(seed, 2)
            emission_weights = jr.normal(rng, shape=(num_emission_dims, num_latent_dims))

        if emission_bias is None:
            emission_bias = np.zeros(num_emission_dims)

        if emission_scale_tril is None:
            emission_scale_tril = 1.0**2 * np.eye(num_emission_dims)

        initial_condition = StandardInitialCondition(initial_mean=initial_state_mean,
                                                     initial_scale_tril=initial_state_scale_tril)
        transitions = StationaryDynamics(weights=dynamics_weights,
                                         bias=dynamics_bias,
                                         scale_tril=dynamics_scale_tril)
        emissions = GaussianEmissions(weights=emission_weights,
                                         bias=emission_bias,
                                         scale_tril=emission_scale_tril)
        super(GaussianLDS, self).__init__(initial_condition,
                                          transitions,
                                          emissions)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(*children)
        return obj

    @property
    def emissions_noise_covariance(self):
        R_sqrt = self._emissions.scale_tril
        return R_sqrt @ R_sqrt.T

    # Methods for inference
    def infer_posterior(self, data):
        """
        Compute the exact posterior by extracting the natural parameters
        of the LDS, namely the block tridiagonal precision matrix (J) and
        the linear coefficient (h).
        """
        # Shorthand names for parameters
        m1 = self.initial_mean
        Q1 = self.initial_covariance
        A = self.dynamics_matrix
        b = self.dynamics_bias
        Q = self.dynamics_noise_covariance
        C = self.emissions_matrix
        d = self.emissions_bias
        R = self.emissions_noise_covariance
        seq_len = data.shape[0]

        # diagonal blocks of precision matrix
        J_diag = np.dot(C.T, np.linalg.solve(R, C))  # from observations
        J_diag = np.tile(J_diag[None, :, :], (seq_len, 1, 1))
        J_diag = J_diag.at[0].add(np.linalg.inv(Q1))
        J_diag = J_diag.at[:-1].add(np.dot(A.T, np.linalg.solve(Q, A)))
        J_diag = J_diag.at[1:].add(np.linalg.inv(Q))

        # lower diagonal blocks of precision matrix
        J_lower_diag = -np.linalg.solve(Q, A)
        J_lower_diag = np.tile(J_lower_diag[None, :, :], (seq_len - 1, 1, 1))

        # linear potential
        h = np.dot(data - d, np.linalg.solve(R, C))  # from observations
        h = h.at[0].add(np.linalg.solve(Q1, m1))
        h = h.at[:-1].add(-np.dot(A.T, np.linalg.solve(Q, b)))
        h = h.at[1:].add(np.linalg.solve(Q, b))

        return LDSPosterior.infer(J_diag, J_lower_diag, h)

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
        states = posterior.mean()
        lps = self.log_probability(states, data) - posterior.log_prob(states)
        return lps

    @format_dataset
    def fit(self, dataset, method="em", rng=None, num_iters=100, tol=1e-4, verbosity=Verbosity.DEBUG):
        r"""Fit the GaussianLDS to a dataset using the specified method.

        Note: because the observations are Gaussian, we can perform exact EM for a GaussianEM
        (i.e. the model is conjugate).

        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{[batch]} , \text{num\_timesteps} , \text{emissions\_dim})`
            method (str, optional): model fit method.
                Must be one of ["em", "laplace_em"]. Defaults to "em".
            rng (jr.PRNGKey, optional): Random seed.
                Defaults to None.
            num_iters (int, optional): number of fit iterations.
                Defaults to 100.
            tol (float, optional): tolerance in log probability to determine convergence.
                Defaults to 1e-4.
            verbosity (Verbosity, optional): print verbosity.
                Defaults to Verbosity.DEBUG.

        Raises:
            ValueError: if fit method is not reocgnized

        Returns:
            elbos (np.ndarray): elbos at each fit iteration
            model (LDS): the fitted model
            posteriors (LDSPosterior): the fitted posteriors
        """
        model = self
        kwargs = dict(num_iters=num_iters, tol=tol, verbosity=verbosity)

        if method == "em":
            elbos, lds, posteriors = em(model, dataset, **kwargs)
        elif method == "laplace_em":
            if rng is None:
                raise ValueError("Laplace EM requires a PRNGKey. Please provide an rng to fit.")
            elbos, lds, posteriors = laplace_em(rng, model, dataset, **kwargs)
        else:
            raise ValueError(f"Method {method} is not recognized/supported.")

        return elbos, lds, posteriors



@register_pytree_node_class
class PoissonLDS(LDS):
    def __init__(self,
                 num_latent_dims,
                 num_emission_dims=None,
                 initial_state_mean=None,
                 initial_state_scale_tril=None,
                 dynamics_weights=None,
                 dynamics_bias=None,
                 dynamics_scale_tril=None,
                 emission_weights=None,
                 emission_bias=None,
                 emission_scale_tril=None,  # TODO: remove
                 seed=None):
        r"""LDS with Poisson emissions.

        .. math::
            p(y_t | x_t) \sim \text{Po}(\lambda = \lambda_{x_t})

        The PoissonLDS can be initialized by specifying each parameter explicitly,
        or you can simply specify the ``num_latent_dims``, ``num_emission_dims``, and ``seed``
        to create a GaussianLDS with generic, randomly initialized parameters.

        Args:
            num_latent_dims (int): number of latent dims
            num_emission_dims (int, optional): number of emissions dims. Defaults to None.
            initial_state_mean (np.ndarray, optional): [description]. Defaults to None.
            initial_state_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            dynamics_weights (np.ndarray, optional): [description]. Defaults to None.
            dynamics_bias (np.ndarray, optional): [description]. Defaults to None.
            dynamics_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            emission_weights (np.ndarray, optional): [description]. Defaults to None.
            emission_bias (np.ndarray, optional): [description]. Defaults to None.
            emission_scale_tril (np.ndarray, optional): [description]. Defaults to None.
            seed (jr.PRNGKey, optional): [description]. Defaults to None.
        """

        if initial_state_mean is None:
            initial_state_mean = np.zeros(num_latent_dims)

        if initial_state_scale_tril is None:
            initial_state_scale_tril = np.eye(num_latent_dims)

        if dynamics_weights is None:
            seed, rng = jr.split(seed, 2)
            dynamics_weights = random_rotation(rng, num_latent_dims, theta=np.pi/20)

        if dynamics_bias is None:
            dynamics_bias = np.zeros(num_latent_dims)

        if dynamics_scale_tril is None:
            dynamics_scale_tril = 0.1**2 * np.eye(num_latent_dims)

        if emission_weights is None:
            seed, rng = jr.split(seed, 2)
            emission_weights = jr.normal(rng, shape=(num_emission_dims, num_latent_dims))

        if emission_bias is None:
            emission_bias = np.zeros(num_emission_dims)

        initial_condition = StandardInitialCondition(initial_mean=initial_state_mean,
                                                     initial_scale_tril=initial_state_scale_tril)
        transitions = StationaryDynamics(weights=dynamics_weights,
                                         bias=dynamics_bias,
                                         scale_tril=dynamics_scale_tril)
        emissions = PoissonEmissions(weights=emission_weights,
                                     bias=emission_bias)
        super(PoissonLDS, self).__init__(initial_condition,
                                          transitions,
                                          emissions)

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._dynamics,
                    self._emissions)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(*children)
        return obj