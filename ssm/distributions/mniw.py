import jax.numpy as np
import jax.random as jr
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates import jax as tfp


class MatrixNormalInverseWishart(tfp.distributions.Distribution):
    r"""
    The matrix-normal inverse-Wishart (MNIW) is a joint distribution on a
    rectangular matrix `A \in \mathbb{R}^{n \times m}` and a positive definite
    matrix `\Sigma \in \mathbb{R}^{n \times n}`. The generative process is,
    ..math::
        A | \Sigma \sim \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0)
            \Sigma \sim \mathrm{IW}(\Sigma | \Psi_0, \nu_0)

    The parameters are:
        `M_0`: the prior mean (location) of `A`
        `V_0`: the prior covariance of the columns of `A`
        `\nu_0`: the prior degrees of freedom for the noise covariance
        `Psi_0`: the prior scale matrix for the noise covariance `\Sigma`

    In the special case where the covariates are always one, `x = 1`, and
    hence the matrices `A` and `M_0` are really just column vectors `a` and
    `\mu_0`, the MNIW reduces to a NIW distribution,
    ..math::
        a \sim \mathrm{NIW}{\mu_0, 1/V_0, \nu_0, \Psi_0}
    (`\kappa_0` is a precision in the NIW, whereas `V_0` is a covariance.)

    The MNIW pdf is proportional to,
    ..math::
        \log p(A , \Sigma) =
            -p/2 \log |\Sigma|
            -1/2 Tr(V_0^{-1} A^\top \Sigma^{-1} A)
               + Tr( V_0^{-1} M_0^\top \Sigma^{-1} A)
            -1/2 Tr(M_0 V_0^{-1} M_0^\top \Sigma^{-1})
            -(\nu_0 + n + 1)/2 \log|\Sigma|
            -1/2 Tr(\Psi_0 \Sigma^{-1})
            + c.

    It is conjugate prior for the weights
    `A` and covariance `Sigma` in a Bayesian multivariate linear regression,
    ..math::
        y | x ~ N(Ax, Sigma)

    Expanding the linear regression likelihood,
    ..math::
        \log p(y | x) =
            -1/2 \log |\Sigma|
            -1/2 Tr((y - Ax)^\top \Sigma^{-1} (y - Ax))
          = -1/2 \log |\Sigma|
            -1/2 Tr(x x^\top A^\top \Sigma^{-1} A)
               + Tr(x y^\top \Sigma^{-1} A)
            -1/2 Tr(y y^\top \Sigma^{-1})
    Its natural parameters are
    .. math::
        \eta_1 = -1/2 A^\top \Sigma^{-1} A
        \eta_2 = \Sigma^{-1} A
        \eta_3 = -1/2 \Sigma^{-1}
    and they correspond to the sufficient statistics,
    .. math::
        t(x)_1 = x x^\top,
        t(x)_2 = y x^\top,
        t(x)_3 = y y^\top,
    Collecting terms, the MNIW prior contributes the following pseudo-counts
    and pseudo-observations,
    ..math::
        n_1 = \nu_0 + n + p + 1
        s_1 = V_0^{-1}
        s_2 = M_0 V_0^{-1}
        s_3 = \Psi_0 + M_0 V_0^{-1} M_0^\top
    We default to an improper prior, with `n_1 = 0` and
     `s_i = 0` for `i=1..3`.
    """
    def __init__(
        self,
        loc,
        column_covariance,
        df,
        scale,
        validate_args=False,
        allow_nan_stats=True,
        name="MatrixNormalInverseWishart",
    ):

        symmetrize = lambda X: 0.5 * (X + np.swapaxes(X, -1, -2))
        self.loc = loc
        self.column_covariance = column_covariance
        self.df = df
        self.scale = symmetrize(scale)

        super(MatrixNormalInverseWishart, self).__init__(
            dtype=loc.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(locals()),
            name=name,
        )

    @property
    def in_dim(self):
        return self.loc.shape[-1]

    @property
    def out_dim(self):
        return self.loc.shape[-2]

    def _log_prob(self, data):
        r"""Compute the prior log probability of LinearRegression weights
        and covariance matrix under this MNIW prior.  The IW pdf is provided
        in scipy.stats.  The matrix normal pdf is,
        ..math::
            \log p(A | M, \Sigma, V) =
                -1/2 Tr \left[ V^{-1} (A - M)^\top \Sigma^{-1} (A - M) \right]
                -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|
              = -1/2 Tr(B B^T) -np/2 \log (2\pi) -n/2 \log |V| -p/2 \log |\Sigma|
        where
        ..math::
            B = U^{-1/2} (A - M) (V^T)^{-1/2}
        """
        weights, covariance_matrix = data

        # Evaluate the matrix normal log pdf
        lp = 0

        # \log p(A | M_0, \Sigma, V_0)
        if np.all(np.isfinite(self.column_covariance)):
            Vsqrt = np.linalg.cholesky(self.column_covariance)
            Ssqrt = np.linalg.cholesky(covariance_matrix)
            B = np.linalg.solve(Ssqrt, np.linalg.solve(
                Vsqrt, (weights - self.loc).T).T)
            lp += -0.5 * np.sum(B**2)
            lp += -self.out_dim * np.sum(np.log(np.diag(Vsqrt)))
            lp += -0.5 * self.in_dim * self.out_dim * np.log(2 * np.pi)
            lp += -self.in_dim * np.sum(np.log(np.diag(Ssqrt)))

        # For comparison, compute the big multivariate normal log pdf explicitly
        # Note: we have to do the kron in the reverse order of what is given
        # on Wikipedia since ravel() is done in row-major ('C') order.
        # lp_test = scipy.stats.multivariate_normal.logpdf(
        #     np.ravel(weights), np.ravel(self.M0),
        #     np.kron(covariance_matrix, self.V0))
        # assert np.allclose(lp, lp_test)

        # \log p(\Sigma | \Psi0, \nu0)
        if self.df >= self.out_dim and \
            np.all(np.linalg.eigvalsh(self.scale) > 0):

            # TODO: get rid of all the inverses
            wish = tfp.distributions.WishartTriL(
                self.df, np.linalg.cholesky(np.linalg.inv(self.scale))
            )
            lp += wish.log_prob(np.linalg.inv(covariance_matrix))

        return lp

    def _mode(self):
        r"""Solve for the mode. Recall,
        .. math::
            p(A, \Sigma) \propto
                \mathrm{N}(\vec(A) | \vec(M_0), \Sigma \kron V_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)
        The optimal mean is :math:`A^* = M_0`. Substituting this in,
        .. math::
            p(A^*, \Sigma) \propto IW(\Sigma | \nu_0 + p, \Psi_0)
        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + p + n + 1)
        """
        A = self.loc
        Sigma = self.scale / (self.df[..., None, None] + self.in_dim + self.out_dim + 1)
        # Sigma += 1e-8 * np.eye(self.out_dim)
        return A, Sigma

