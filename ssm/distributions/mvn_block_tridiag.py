
import jax.numpy as np
import jax.random as jr
from jax import lax, value_and_grad
from jax.scipy.linalg import solve_triangular
from tensorflow_probability.python import internal

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization
# from tensorflow_probability.python.internal import parameter_properties
# from tensorflow_probability.python.internal import prefer_static as ps
# from tfp.bijectors import fill_scale_tril as fill_scale_tril_bijector
# from tensorflow_probability.python.internal import dtype_util


def _forward_pass(J_diag, J_lower_diag, h, logc):
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
    (_, _, log_Z), (filtered_Js, filtered_hs) = lax.scan(marginalize, (Jp0, hp0, logc), np.arange(num_timesteps))
    return log_Z, (filtered_Js, filtered_hs)


class MultivariateNormalBlockTridiag(tfp.distributions.Distribution):
    """
    Multivariate normal distribution whose _precision_ matrix is
    block tridiagonal.

        x ~ N(\mu, \Sigma)

    where

        \Sigma^{-1} = J = [[J_{0,0},   J_{0,1},   0,       0,      0],
                           [J_{1,0},   J_{1,1},   J_{1,2}, 0,      0],
                           [0,         J_{2,1},   J_{2,2}, \ddots, 0],
                           [0,         0,         \ddots,  \ddots,  ],

    is block tridiagonal, and J_{t, t+1} = J_{t+1, t}^T.

    This will serve as a posterior distribution over latent states of an LDS.

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
                 precision_diag_blocks=None,
                 precision_lower_diag_blocks=None,
                 linear_potential=None,
                 logc=None,
                 mean=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="MultivariateNormalBlockTridiag",
             ) -> None:
        assert precision_diag_blocks is not None
        assert precision_diag_blocks.ndim == 3
        # TODO: extend to have batches?
        self._num_timesteps, self._dim = precision_diag_blocks.shape[:2]
        assert precision_diag_blocks.shape[2] == self._dim
        self._precision_diag_blocks = precision_diag_blocks

        assert precision_lower_diag_blocks is not None
        assert precision_lower_diag_blocks.shape == (self._num_timesteps - 1, self._dim, self._dim)
        self._precision_lower_diag_blocks = precision_lower_diag_blocks

        assert mean is not None or linear_potential is not None
        if mean is not None and linear_potential is None:
            # Convert mean (\mu) to linear potential (h)
            # using precision blocks (J) via h = J \mu
            raise NotImplementedError

        else:
            assert linear_potential.shape == (self._num_timesteps, self._dim)
            self._linear_potential = linear_potential

        self._logc = logc

        # just get log normalizer (with logc)
        # self._log_noramlizer, _ = _forward_pass(self._precision_diag_blocks,
        #                                         self._precision_lower_diag_blocks,
        #                                         self._linear_potential,
        #                                         self._logc)

        self.ran_message_passing = False

        super().__init__(
            dtype=precision_diag_blocks.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(precision_diag_blocks=self._precision_diag_blocks,
                            precision_lower_diag_blocks=self._precision_lower_diag_blocks,
                            linear_potential=self._linear_potential,
                            logc=self._logc),
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            precision_diag_blocks=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            precision_lower_diag_blocks=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            linear_potential=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
            logc=tfp.internal.parameter_properties.ParameterProperties(event_ndims=0),
        )

    def _log_prob(self, data, **kwargs):
        lp = -0.5 * np.einsum('...ti,tij,...tj->...', data, self._precision_diag_blocks, data)
        lp += -np.einsum('...ti,tij,...tj->...', data[1:], self._precision_lower_diag_blocks, data[:-1])
        lp += np.einsum('...ti,ti->...', data, self._linear_potential)
        lp -= self.log_normalizer
        return lp

        # log p(x,y) - log p(y) = log p(x|y)

    def message_passing(self):
        # Run message passing code to get the log normalizer, the filtering potentials,
        # and the expected values of x.
        f = value_and_grad(_forward_pass, argnums=(0, 1, 2), has_aux=True)
        (self._log_normalizer, (self._filtered_Js, self._filtered_hs)), grads = \
            f(self._precision_diag_blocks,
              self._precision_lower_diag_blocks,
              self._linear_potential,
              self._logc)

        self._ExxT = -2 * grads[0]
        self._ExnxT = -grads[1]
        self._Ex = grads[2]

        self.ran_message_passing = True

    @property
    def log_normalizer(self):
        if not self.ran_message_passing:
            self.message_passing()
        return self._log_normalizer

    @property
    def mean(self):
        if not self.ran_message_passing:
            self.message_passing()
        return self._Ex

    @property
    def second_moments(self):
        if not self.ran_message_passing:
            self.message_passing()
        return self._ExxT, self._ExnxT

    def _sample(self, seed=None, sample_shape=()):
        filtered_Js = self._filtered_Js
        filtered_hs = self._filtered_hs
        J_lower_diag = self._precision_lower_diag_blocks

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
        _, samples = lax.scan(_step, (x_T, seed), inputs)
        return samples

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ex = self.mean
        ExxT, ExnxT = self.second_moments
        J_diag = self._precision_diag_blocks
        J_lower_diag = self._precision_lower_diag_blocks
        h = self._linear_potential

        entropy = 0.5 * np.sum(J_diag * ExxT)
        entropy += np.sum(J_lower_diag * ExnxT)
        entropy -= np.sum(h, Ex)
        entropy += self.log_normalizer
        return entropy
