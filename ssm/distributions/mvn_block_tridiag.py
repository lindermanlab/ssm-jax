import jax.numpy as np
import jax.random as jr
from jax import lax, value_and_grad, vmap
from jax.scipy.linalg import solve_triangular

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization
tfd = tfp.distributions


def block_tridiag_mvn_log_normalizer(J_diag, J_lower_diag, h):
    """ TODO
    """
    # extract dimensions
    num_timesteps, dim = J_diag.shape[:2]

    # Pad the L's with one extra set of zeros for the last predict step
    J_lower_diag_pad = np.concatenate((J_lower_diag, np.zeros((1, dim, dim))), axis=0)

    def marginalize(carry, t):
        Jp, hp, lp = carry

        # Condition
        Jc = J_diag[t] + Jp
        hc = h[t] + hp

        # Predict -- Cholesky approach seems unstable!
        sqrt_Jc = np.linalg.cholesky(Jc)
        trm1 = solve_triangular(sqrt_Jc, hc, lower=True)
        trm2 = solve_triangular(sqrt_Jc, J_lower_diag_pad[t].T, lower=True)
        log_Z = 0.5 * dim * np.log(2 * np.pi)
        log_Z += -np.sum(np.log(np.diag(sqrt_Jc)))  # sum these terms only to get approx log|J|
        log_Z += 0.5 * np.dot(trm1.T, trm1)
        Jp = -np.dot(trm2.T, trm2)
        hp = -np.dot(trm2.T, trm1)

        # Alternative predict step:
        # log_Z = 0.5 * dim * np.log(2 * np.pi)
        # log_Z += -0.5 * np.linalg.slogdet(Jc)[1]
        # log_Z += 0.5 * np.dot(hc, np.linalg.solve(Jc, hc))
        # Jp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, J_lower_diag_pad[t].T))
        # hp = -np.dot(J_lower_diag_pad[t], np.linalg.solve(Jc, hc))

        new_carry = Jp, hp, lp + log_Z
        return new_carry, (Jc, hc)

    # Initialize
    Jp0 = np.zeros((dim, dim))
    hp0 = np.zeros((dim,))
    (_, _, log_Z), (filtered_Js, filtered_hs) = lax.scan(marginalize, (Jp0, hp0, 0), np.arange(num_timesteps))
    return log_Z, (filtered_Js, filtered_hs)


# @register_pytree_node_class
class MultivariateNormalBlockTridiag(tfd.Distribution):
    """
    The Gaussian linear dynamical system's posterior distribution over latent states
    is a multivariate normal distribution whose _precision_ matrix is
    block tridiagonal.

        x | y ~ N(\mu, \Sigma)

    where

        \Sigma^{-1} = J = [[J_{0,0},   J_{0,1},   0,       0,      0],
                           [J_{1,0},   J_{1,1},   J_{1,2}, 0,      0],
                           [0,         J_{2,1},   J_{2,2}, \ddots, 0],
                           [0,         0,         \ddots,  \ddots,  ],

    is block tridiagonal, and J_{t, t+1} = J_{t+1, t}^T.

    The pdf is

        p(x) = exp \{-1/2 x^T J x + x^T h - \log Z(J, h) \}
             = exp \{- 1/2 \sum_{t=1}^T x_t^T J_{t,t} x_t
                     - \sum_{t=1}^{T-1} x_{t+1}^T J_{t+1,t} x_t
                     + \sum_{t=1}^T x_t^T h_t
                     -\log Z(J, h)\}

    where J = \Sigma^{-1} and h = \Sigma^{-1} \mu = J \mu.

    Using exponential family tricks we know that

        E[x_t] = \grad_{h_t} \log Z(J, h)
        E[x_t x_t^T] = -2 \grad_{J_{t,t}} \log Z(J, h)
        E[x_{t+1} x_t^T] = -\grad_{J_{t+1,t}} \log Z(J, h)

    These are the expectations we need for EM.
    """
    def __init__(self,
                 precision_diag_blocks,
                 precision_lower_diag_blocks,
                 linear_potential,
                 log_normalizer,
                 filtered_precisions,
                 filtered_linear_potentials,
                 expected_states,
                 expected_states_squared,
                 expected_states_next_states,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="MultivariateNormalBlockTridiag",
             ) -> None:

        self._precision_diag_blocks = precision_diag_blocks
        self._precision_lower_diag_blocks = precision_lower_diag_blocks
        self._linear_potential = linear_potential
        self._log_normalizer = log_normalizer
        self._filtered_precisions = filtered_precisions
        self._filtered_linear_potentials = filtered_linear_potentials
        self._expected_states = expected_states
        self._expected_states_squared = expected_states_squared
        self._expected_states_next_states = expected_states_next_states

        # We would detect the dtype dynamically but that would break vmap
        # see https://github.com/tensorflow/probability/issues/1271
        dtype = np.float32
        super(MultivariateNormalBlockTridiag, self).__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(precision_diag_blocks=self._precision_diag_blocks,
                            precision_lower_diag_blocks=self._precision_lower_diag_blocks,
                            linear_potential=self._linear_potential,
                            log_normalizer=self._log_normalizer,
                            filtered_precisions=self._filtered_precisions,
                            filtered_linear_potentials=self._filtered_linear_potentials,
                            expected_states=self._expected_states,
                            expected_states_squared=self._expected_states_squared,
                            expected_states_next_states=self._expected_states_next_states),
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            precision_diag_blocks=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
            precision_lower_diag_blocks=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
            linear_potential=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            log_normalizer=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            filtered_precisions=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
            filtered_linear_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            expected_states_squared=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
            expected_states_next_states=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
        )

    @classmethod
    def infer(cls,
              precision_diag_blocks,
              precision_lower_diag_blocks,
              linear_potential):
        assert precision_diag_blocks.ndim == 3
        num_timesteps, dim = precision_diag_blocks.shape[:2]
        assert precision_diag_blocks.shape[2] == dim
        assert precision_lower_diag_blocks.shape == (num_timesteps - 1, dim, dim)
        assert linear_potential.shape == (num_timesteps, dim)

        # Run message passing code to get the log normalizer, the filtering potentials,
        # and the expected values of x. Technically, the natural parameters are -1/2 J
        # so we need to do a little correction of the gradients to get the expectations.
        f = value_and_grad(block_tridiag_mvn_log_normalizer, argnums=(0, 1, 2), has_aux=True)
        (log_normalizer, (filtered_precisions, filtered_linear_potentials)), grads = \
            f(precision_diag_blocks, precision_lower_diag_blocks, linear_potential)

        # Correct for the -1/2 J -> J implementation
        expected_states_squared = -2 * grads[0]
        expected_states_next_states = -grads[1]
        expected_states = grads[2]

        return cls(precision_diag_blocks,
                   precision_lower_diag_blocks,
                   linear_potential,
                   log_normalizer,
                   filtered_precisions,
                   filtered_linear_potentials,
                   expected_states,
                   expected_states_squared,
                   expected_states_next_states)

    @classmethod
    def infer_from_precision_and_mean(cls,
                                      precision_diag_blocks,
                                      precision_lower_diag_blocks,
                                      mean):
        assert precision_diag_blocks.ndim == 3
        num_timesteps, dim = precision_diag_blocks.shape[:2]
        assert precision_diag_blocks.shape[2] == dim
        assert precision_lower_diag_blocks.shape == (num_timesteps - 1, dim, dim)
        assert mean.shape == (num_timesteps, dim)

        # Convert the mean to the linear potential
        linear_potential = np.einsum('tij,tj->ti', precision_diag_blocks, mean)
        linear_potential = linear_potential.at[:-1].add(
            np.einsum('tji,tj->ti', precision_lower_diag_blocks, mean[1:]))
        linear_potential = linear_potential.at[1:].add(
            np.einsum('tij,tj->ti', precision_lower_diag_blocks, mean[:-1]))

        # Call the constructor above
        return cls.infer(precision_diag_blocks,
                         precision_lower_diag_blocks,
                         linear_potential)

    # Properties to get private class variables
    @property
    def precision_diag_blocks(self):
        return self._precision_diag_blocks

    @property
    def precision_lower_diag_blocks(self):
        return self._precision_lower_diag_blocks

    @property
    def linear_potential(self):
        return self._linear_potential

    @property
    def log_normalizer(self):
        return self._log_normalizer

    @property
    def filtered_precisions(self):
        return self._filtered_precisions

    @property
    def filtered_linear_potentials(self):
        return self._filtered_linear_potentials

    @property
    def expected_states(self):
        return self._expected_states

    @property
    def expected_states_squared(self):
        return self._expected_states_squared

    @property
    def expected_states_next_states(self):
        return self._expected_states_next_states

    def _log_prob(self, data, **kwargs):
        lp = -0.5 * np.einsum('...ti,tij,...tj->...', data, self._precision_diag_blocks, data)
        lp += -np.einsum('...ti,tij,...tj->...', data[1:], self._precision_lower_diag_blocks, data[:-1])
        lp += np.einsum('...ti,ti->...', data, self._linear_potential)
        lp -= self.log_normalizer
        return lp

    def _mean(self):
        return self.expected_states

    def _covariance(self):
        """
        NOTE: This computes the _marginal_ covariance Cov[x_t] for each t
        """
        Ex = self._expected_states
        ExxT = self._expected_states_squared
        return ExxT - np.einsum("...i,...j->...ij", Ex, Ex)

    def _sample(self, seed=None, sample_shape=()):
        filtered_Js = self._filtered_precisions
        filtered_hs = self._filtered_linear_potentials
        J_lower_diag = self._precision_lower_diag_blocks

        def sample_single(seed, filtered_Js, filtered_hs, J_lower_diag):

            def _sample_info_gaussian(seed, J, h, shape):
                # TODO: avoid inversion. see https://github.com/mattjj/pybasicbayes/blob/master/pybasicbayes/util/stats.py#L117-L122
                # L = np.linalg.cholesky(J)
                # x = np.random.randn(h.shape[0])
                # return scipy.linalg.solve_triangular(L,x,lower=True,trans='T') \
                #     + dpotrs(L,h,lower=True)[0]
                cov = np.linalg.inv(J)
                loc = np.einsum("...ij,...j->...i", cov, h)
                return tfp.distributions.MultivariateNormalFullCovariance(
                    loc=loc, covariance_matrix=cov).sample(sample_shape=shape, seed=seed)

            def _step(carry, inpt):
                x_next, seed = carry
                Jf, hf, L = inpt

                # Condition on the next observation
                Jc = Jf
                hc = hf - x_next @ L

                # Split the seed
                seed, this_seed = jr.split(seed)
                x = _sample_info_gaussian(this_seed, Jc, hc, sample_shape)
                return (x, seed), x

            # Initialize with sample of last state
            seed_T, seed = jr.split(seed)
            x_T = _sample_info_gaussian(seed_T, filtered_Js[-1], filtered_hs[-1], sample_shape)
            inputs = (filtered_Js[:-1][::-1], filtered_hs[:-1][::-1], J_lower_diag[::-1])
            _, x_rev = lax.scan(_step, (x_T, seed), inputs)
            return np.concatenate((x_rev[::-1], x_T[None, ...]), axis=0)

        # batch mode
        if filtered_Js.ndim == 4:
            samples = vmap(sample_single)(seed, filtered_Js, filtered_hs, J_lower_diag)

        # non-batch mode
        else:
            samples = sample_single(seed, filtered_Js, filtered_hs, J_lower_diag)
        return samples

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ex = self._expected_states
        ExxT = self._expected_states_squared
        ExnxT = self._expected_states_next_states
        J_diag = self._precision_diag_blocks
        J_lower_diag = self._precision_lower_diag_blocks
        h = self._linear_potential

        entropy = 0.5 * np.sum(J_diag * ExxT)
        entropy += np.sum(J_lower_diag * ExnxT)
        entropy -= np.sum(h, Ex)
        entropy += self.log_normalizer
        return entropy

    def tree_flatten(self):
        children = (self._precision_diag_blocks,
                    self._precision_lower_diag_blocks,
                    self._linear_potential,
                    self._log_normalizer,
                    self._filtered_precisions,
                    self._filtered_linear_potentials,
                    self._expected_states,
                    self._expected_states_squared,
                    self._expected_states_next_states)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
