
import jax.numpy as np
from jax import lax

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization

from ssm.distributions.mniw import MatrixNormalInverseWishart
tfb = tfp.bijectors
tfd = tfp.distributions

from ssm.distributions.expfam import ExponentialFamilyDistribution


class GaussianLinearRegression(ExponentialFamilyDistribution,
                               tfp.distributions.Distribution):
    """Linear regression with Gaussian noise.
    """
    def __init__(
        self,
        weights,
        bias,
        scale_tril,
        validate_args=False,
        allow_nan_stats=True,
        name="GaussianLinearRegression",
    ):
        self._weights = weights
        self._bias = bias
        self._scale_tril = scale_tril

        super(GaussianLinearRegression, self).__init__(
            dtype=weights.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(locals()),
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            weights=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            bias=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            scale_tril=tfp.internal.parameter_properties.ParameterProperties(
                event_ndims=2,
                # shape_fn=lambda sample_shape: ps.concat([sample_shape, sample_shape[-1:]], axis=0),
                # default_constraining_bijector_fn=lambda: tfp.bijectors.fill_scale_tril.FillScaleTriL(diag_shift=dtype_util.eps(dtype)))
        ))

    @property
    def data_dimension(self):
        return self.weights.shape[-2]

    @property
    def covariate_dimension(self):
        return self.weights.shape[-1]

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def covariance(self):
        return np.einsum('...ij,...ji->...ij', self.scale_tril, self.scale_tril)

    @property
    def scale_tril(self):
        return self._scale_tril

    def predict(self, covariates=None):
        mean = np.einsum('...i,...ji->...j', covariates, self.weights) + self.bias
        return tfp.distributions.MultivariateNormalTriL(mean, self.scale_tril)

    def _event_shape(self):
        return self.bias.shape[-1]

    def _sample_n(self, n, covariates=None, seed=None):
        return self.predict(covariates).sample(sample_shape=n, seed=seed)

    def _log_prob(self, data, covariates=None, **kwargs):
        return self.predict(covariates).log_prob(data)

    def expected_log_prob(self,
                          expected_data,
                          expected_covariates,
                          expected_data_squared,
                          expected_data_covariates,
                          expected_covariates_squared):
        """
        Helper function to compute the expected log probability
        under a Gaussian distribution (x, y) with given expectations.

        We have
        ..math:
            E_q[\log p(y \mid x)] =
                \langle -1/2 \Sigma^{-1}, E_q[yy^\top] \rangle +
                \langle \Sigma^{-1}W, E_q[yx^\top] \rangle +
                \langle -1/2 W^\top \Sigma^{-1} W, E_q[xx^\top] \rangle +
                \langle \Sigma^{-1}b, E_q[y] \rangle +
                \langle -W^\top \Sigma^{-1}b, E_q[x] \rangle +
                -1/2 \log |\Sigma| -1/2 b^\top \Sigma^{-1} b
        """
        transpose = lambda x: np.swapaxes(x, -1, -2)
        Si = np.linalg.inv(self.covariance)
        SiW = np.einsum('...ij,...jk->...ik', Si, self.weights)
        Sib = np.einsum('...ij,...j->...i', Si, self.bias)
        WTSiW = np.einsum('...ij,...jk->...ik', transpose(self.weights), SiW)
        WTSib = np.einsum('...ij,...j->...i', transpose(self.weights), Sib)
        bTSib = np.einsum('...i,...i->...', self.bias, Sib)


        ell = np.sum(-0.5 * Si * expected_data_squared, axis=(-1, -2))
        ell += np.sum(SiW * expected_data_covariates, axis=(-1, -2))
        ell += np.sum(-0.5 * WTSiW * expected_covariates_squared, axis=(-1, -2))
        ell += np.sum(Sib * expected_data, axis=-1)
        ell += np.sum(-WTSib * expected_covariates, axis=-1)
        ell += -0.5 * np.linalg.slogdet(self.covariance)[1]
        ell += -0.5 * bTSib
        ell += -0.5 * self.data_dimension * np.log(2 * np.pi)
        return ell

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    @staticmethod
    def sufficient_statistics(data, covariates):
        return (1.0,
                np.outer(covariates, covariates),
                covariates,
                1.0,
                np.outer(data, covariates),
                data,
                np.outer(data, data))


class GaussianLinearRegressionPrior(MatrixNormalInverseWishart):
    r"""The conjugate prior for a Gaussian linear regression is a matrix
    normal inverse Wishart (MNIW), but it's helpful to parameterize
    a slightly different prior for the bias and the regression weights
    since we often want the bias to be only weakly penalized.

    Let y \in \mathbb{R}^n and x \in \mathbb{R}^p. The likelihood is:
        .. math:
            y \sim N(A x + b, \Sigma)
    Let W = [A, b] \in \mathbb{R}^{n \times p + 1} denote the combined
    weights and biases.

    The prior on (W, \Sigma) is
        .. math
            \Sigma         \sim \mathrm{IW}(\nu, \Psi)
            W \mid \Sigma  \sim \mathrm{MN}(M, \Sigma, V)

    where \nu is the degrees of freedom, \Psi is the scale of the inverse
    Wishart distribution, M is the mean of the weights, and V is the
    covariance of the columns of W.
    """
    def __repr__(self) -> str:
        return "<GaussianLinearRegressionPrior batch_shape={} event_shape={}>".\
            format(self._loc.shape[:-2], self._loc.shape[-2:])

    def _mode(self):
        r"""Solve for the mode. Recall,
        .. math::
            p(\mu, \Sigma) \propto
                \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)
        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        .. math::
            p(\mu^*, \Sigma) \propto IW(\Sigma | \nu_0 + 1, \Psi_0)
        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + d + 2)
        """
        weights_and_bias = self._loc
        scale = np.einsum("...,...ij->...ij", 1 / (self._df + self.dim[0] + self.dim[1] + 2), self._scale)
        return dict(weights=weights_and_bias[...,:,:-1],
                    bias=weights_and_bias[...,:,-1],
                    scale_tril=np.linalg.cholesky(scale))

    @property
    def natural_parameters(self):
        """Compute pseudo-observations from standard NIW parameters."""
        T = lambda X: np.swapaxes(X, -1, -2)
        row_dim, col_dim = self.dim
        Vi = np.linalg.inv(self._scale_column)
        MVi = T(Vi @ T(self._loc))
        MViMT = MVi @ T(self._loc)

        s1 = self._df + row_dim + col_dim + 1    # 1
        s2 = Vi[...,:-1,:-1]                    # xx^T
        s3 = Vi[...,:-1,-1]                     # x
        s4 = Vi[...,-1,-1]                      # 1
        s5 = MVi[...,:,:-1]                     # yx^T
        s6 = MVi[...,:, -1]                     # y
        s7 = self._scale + MViMT                 # yy^T
        return s1, s2, s3, s4, s5, s6, s7

    @classmethod
    def from_natural_parameters(cls, natural_params):
        s1, s2, s3, s4, s5, s6, s7 = natural_params

        T = lambda X: np.swapaxes(X, -1, -2)
        out_dim = s6.shape[-1]
        in_dim = s3.shape[-1] + 1
        df = s1 - out_dim - in_dim - 1

        # Pad the sufficient statistics to include the bias
        Vi = np.concatenate([np.concatenate([s2,               s3[..., :, None]],   axis=-1),
                             np.concatenate([s3[..., None, :], s4[..., None, None]], axis=-1)],
                             axis=-2)
        MVi = np.concatenate([s5, s6[..., None]], axis=-1)

        V = np.linalg.inv(Vi + 1e-4 * np.eye(in_dim))
        # M = MVi @ V
        M = T(np.linalg.solve(Vi + 1e-4 * np.eye(in_dim), T(MVi)))
        Psi = s7 - M @ Vi @ T(M) + 1e-4 * np.eye(out_dim)
        return cls(M, V, df, Psi)
