import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from ssm.distributions.expfam import ConjugatePrior


class NormalInverseWishart(ConjugatePrior, tfd.JointDistributionNamed):
    def __init__(self, loc, mean_precision, df, scale, **kwargs):
        """
        A normal inverse Wishart (NIW) distribution with

        TODO: Finish this description
        Args:
            loc:            \mu_0 in math above
            mean_precision: \kappa_0
            df:             \nu
            scale:          \Psi
        """
        # Store hyperparameters.
        # Note: these should really be private.
        self.loc = loc
        self.mean_precision = mean_precision
        self.df = df
        self.scale = scale

        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        # Note: this could be done more efficiently.
        self.wishart_scale_tril = np.linalg.cholesky(np.linalg.inv(scale))

        super(NormalInverseWishart, self).__init__(dict(
            Sigma=lambda: tfd.TransformedDistribution(
                tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                tfb.Chain([tfb.CholeskyOuterProduct(),
                        tfb.CholeskyToInvCholesky(),
                        tfb.Invert(tfb.CholeskyOuterProduct())
                        ])),
            mu=lambda Sigma: tfd.MultivariateNormalFullCovariance(
                loc, Sigma / mean_precision)
        ))

    # These functions compute the pseudo-observations implied by the NIW prior
    # and convert sufficient statistics to a NIW posterior. We'll describe them
    # in more detail below.
    @property
    def natural_parameters(self):
        """Compute pseudo-observations from standard NIW parameters."""
        dim = self.loc.shape[-1]
        s1 = self.df + dim + 2
        s2 = np.einsum('...,...i->...i', self.mean_precision, self.loc)
        s3 = self.scale + self.mean_precision * np.einsum("...i,...j->...ij", self.loc, self.loc)
        s4 = self.mean_precision
        return s1, s2, s3, s4

    @classmethod
    def from_natural_parameters(cls, natural_params):
        """Convert natural parameters into standard parameters and construct."""
        s1, s2, s3, s4 = natural_params
        dim = s2.shape[-1]
        df = s1 - dim - 2
        mean_precision = s4
        # loc = lax.cond(mean_precision > 0,
        #             lambda x: s_2 / mean_precision,
        #             lambda x: np.zeros_like(s_2),
        #             None)
        loc = np.einsum("...i,...->...i", s2, 1 / mean_precision)
        scale = s3 - np.einsum("...,...i,...j->...ij", mean_precision, loc, loc)
        return cls(loc, mean_precision, df, scale)

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
        dim = self.loc.shape[-1]
        covariance = np.einsum("...,...ij->...ij", 1 / (self.df + dim + 2), self.scale)
        return self.loc, covariance
