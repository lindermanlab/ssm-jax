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

        Returns:
            A tfp.JointDistribution object.
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
        chi_1 = self.df + dim + 2
        chi_2 = np.einsum('...,...i->...i', self.mean_precision, self.loc)
        chi_3 = self.scale + \
            self.mean_precision * np.einsum("...i,...j->...ij", self.loc, self.loc)
        chi_4 = self.mean_precision
        return chi_1, chi_2, chi_3, chi_4

    @classmethod
    def from_natural_parameters(cls, natural_params):
        """Convert natural parameters into standard parameters and construct."""
        chi_1, chi_2, chi_3, chi_4 = natural_params
        dim = chi_2.shape[-1]
        df = chi_1 - dim - 2
        mean_precision = chi_4
        loc = np.einsum('..., ...i->...i', 1 / mean_precision, chi_2)
        scale = chi_3 - mean_precision * np.einsum('...i,...j->...ij', loc, loc)
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
        covariance = np.einsum(
            "...,...ij->...ij", 1 / (self.df + self.dim + 2), self.scale
        )
        return self.loc, covariance
