import jax.numpy as np
from jax import lax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tf = tfp.tf2jax

from ssm.distributions.expfam import ConjugatePrior


class MatrixNormalInverseWishart(ConjugatePrior, tfd.JointDistributionNamed):
    def __init__(self, loc, scale_column, df, scale_covariance, **kwargs):
        """
        A matrix normal inverse Wishart (NIW) distribution with
        """
        # Store hyperparameters.
        # Note: these should really be private.
        self._loc = loc
        self._scale_column = scale_column
        self._df = df
        self._scale = scale_covariance

        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        # Note: this could be done more efficiently.
        self.wishart_scale_tril = np.linalg.cholesky(np.linalg.inv(scale_covariance))

        super(MatrixNormalInverseWishart, self).__init__(dict(
            Sigma=lambda: tfd.TransformedDistribution(
                tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                tfb.Chain([tfb.CholeskyOuterProduct(),
                        tfb.CholeskyToInvCholesky(),
                        tfb.Invert(tfb.CholeskyOuterProduct())
                        ])),
            mu=lambda Sigma: tfd.MatrixNormalLinearOperator(
                loc,
                scale_row=tf.linalg.LinearOperatorFullMatrix(Sigma),
                scale_column=tf.linalg.LinearOperatorFullMatrix(scale_column))
        ))

        # Replace the default JointDistributionNamed parameters with the NIW ones
        # because the JointDistributionNamed parameters contain lambda functions,
        # which are not jittable.
        self._parameters = dict(
            loc=loc,
            scale_column=scale_column,
            df=df,
            scale=scale_covariance
        )

    def __repr__(self) -> str:
        return "<MatrixNormalInverseWishart batch_shape={} event_shape={}>".\
            format(self._loc.shape[:-2], self._loc.shape[-2:])

    @property
    def dim(self):
        return self._loc.shape[-2:]

    def _mode(self):
        covariance = np.einsum("...,...ij->...ij",
                               1 / (self._df + sum(self.dim) + 1), self._scale)
        return self._loc, covariance

    @property
    def natural_parameters(self):
        """Compute pseudo-observations from standard NIW parameters."""
        row_dim, col_dim = self.dim
        V0iM0T = np.linalg.solve(self._scale_column, self._loc.T)
        s1 = self._df + row_dim + col_dim + 1
        s2 = np.linalg.inv(self._scale_column)
        s3 = np.swapaxes(V0iM0T, -1, -2)
        s4 = self._scale + self._loc @ V0iM0T
        return s1, s2, s3, s4

    @classmethod
    def from_natural_parameters(cls, natural_params):
        """Convert natural parameters into standard parameters and construct."""
        s1, s2, s3, s4 = natural_params
        row_dim, col_dim = s3.shape[-2:]

        nu0 = s1 - row_dim - col_dim - 1
        def _null_stats(operand):
            V0 = 1e16 * np.eye(col_dim)
            M0 = np.zeros_like(s3)
            Psi0 = np.eye(row_dim)
            return V0, M0, Psi0

        def _stats(operand):
            # TODO: Use Cholesky factorization for these two steps
            V0 = np.linalg.inv(s2 + 1e-16 * np.eye(col_dim))
            M0 = s3 @ V0
            Psi0 = s4 - M0 @ s2 @ M0.T
            return V0, M0, Psi0

        V0, M0, Psi0 = lax.cond(np.allclose(s1, 0), _null_stats, _stats, operand=None)
        return cls(M0, V0, nu0, Psi0)
