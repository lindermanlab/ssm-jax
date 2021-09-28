
import jax.numpy as np

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization
# from tensorflow_probability.python.internal import parameter_properties
# from tensorflow_probability.python.internal import prefer_static as ps
# from tfp.bijectors import fill_scale_tril as fill_scale_tril_bijector
# from tensorflow_probability.python.internal import dtype_util


def _forward_pass(J_diag, J_lower_diag, h):
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
        log_Z += -np.sum(np.log(np.diag(sqrt_Jc)))
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
    return log_Z, filtered_Js, filtered_hs


class MultivariateNormalBlockTridiag(tfp.distributions.Distribution):
    """
    Multivariate normal distribution whose _precision_ matrix is
    block tridiagonal.

        x ~ N(\mu, \Sigma)

    where

        \Sigma^{-1} = J = [[J[0, 0],   J[0, 1],  0, ...         , 0],
                           [J[0, 1].T, J[1, 1],  J[1, 2], 0, ..., 0],
                           [0,         J[1, 2].T J[2, 2], \ddots,  ],

    This will serve as a posterior distribution over latent states of an LDS.

    pdf:

        exp{-1/2 x^T J x + x^T h - log Z(J, h)}

    where



    using exponential family tricks we know that

        E[x] = \grad_h log Z(J, h)

    """
    def __init__(self,
                 precision_diag_blocks=None,
                 precision_lower_diag_blocks=None,
                 linear_potential=None,
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

        # Run message passing code to get the log normalizer, the filtering potentials,
        # and the expected values of x.
        f = value_and_grad(_forward_pass, argnums=(0, 1, 2), has_aux=True)
        (self._log_normalizer, self._filtered_Js, self._filtered_hs), grads = \
            f(self._precision_diag_blocks,
              self._precision_lower_diag_blocks,
              self._linear_potential)

        self._ExxT = -2 * grads[0]
        self._ExnxT = -grads[1]
        self._Ex = grads[2]

        super(MultivariateNormalBlockTridiag, self).__init__(
            dtype=precision_diag_blocks.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(locals()),
            name=name,
        )

    def _log_prob(self, data, **kwargs):
        # todo: compute these quadratic terms
        lp = -0.5 * np.einsum('...ti,tij,...tj->...', data, self._precision_diag_blocks, data)
        lp += -np.einsum('...ti,tij,...tj->...', data[1:], self._precision_lower_diag_blocks, data[:-1])
        lp += np.einsum('...ti,ti->...', data, self._linear_potential)
        lp -= self._log_normalizer
        return lp

    def _mean(self):
        return self._Ex

    def _sample(self, covariates=None, seed=None, sample_shape=()):
        # Run the forward filtering backward sampling algorithm
        # see Yixiu's SVAE-LDS code
        
        # TODO @slinderman: finish porting this code!
        # T, D = h_obs.shape

        # # Run the forward filter
        # log_Z, filtered_Js, filtered_hs = \
        #     _kalman_info_filter(J_ini, h_ini, log_Z_ini,
        #                     J_dyn_11, J_dyn_21, J_dyn_22, h_dyn_1, h_dyn_2, log_Z_dyn,
        #                     J_obs, h_obs, log_Z_obs)

        # # Allocate output arrays
        # samples = np.zeros((T, D))
        # noise = npr.randn(T, D)

        # # Initialize with samples of the last state
        # samples[-1] = _sample_info_gaussian(filtered_Js[-1], filtered_hs[-1], noise[-1])

        # # Run the Kalman information filter
        # for t in range(T-2, -1, -1):
        #     # Extract blocks of the dynamics potentials
        #     J_11 = J_dyn_11[t] if J_dyn_11.shape[0] == T-1 else J_dyn_11[0]
        #     J_21 = J_dyn_21[t] if J_dyn_21.shape[0] == T-1 else J_dyn_21[0]
        #     h_1 = h_dyn_1[t] if h_dyn_1.shape[0] == T-1 else h_dyn_1[0]

        #     # Condition on the next observation
        #     J_post = filtered_Js[t] + J_11
        #     h_post = filtered_hs[t] + h_1 - np.dot(J_21.T, samples[t+1])
        #     samples[t] = _sample_info_gaussian(J_post, h_post, noise[t])

        # return samples
        raise NotImplementedError


    def _entropy(self):
        raise NotImplementedError