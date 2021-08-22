import jax.numpy as np
import jax.random as jr
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.substrates import jax as tfp


class NormalInverseWishart(tfp.distributions.Distribution):
    def __init__(
        self,
        loc,
        mean_precision,
        df,
        scale,
        validate_args=False,
        allow_nan_stats=True,
        name="NormalInverseWishart",
    ):

        self.loc = loc
        self.mean_precision = mean_precision
        self.df = df
        self.scale = scale

        super(NormalInverseWishart, self).__init__(
            dtype=loc.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(locals()),
            name=name,
        )

    @property
    def dim(self):
        return self.loc.shape[-1]

    def _log_prob(self, data):
        """Compute the prior log probability of a MultivariateNormal
        distribution's parameters under this NIW prior.
        Note that the NIW prior is only properly specified in certain
        parameter regimes (otherwise the density does not normalize).
        Only compute the log prior if there is a density.
        """
        mu, Sigma = data

        lp = 0

        if self.df >= self.dim:
            # TODO: get rid of all the inverses
            wish = tfp.distributions.WishartTriL(
                self.df, np.linalg.cholesky(np.linalg.inv(self.scale))
            )
            lp += wish.log_prob(np.linalg.inv(Sigma))

        if self.mean_precision > 0:
            mvn = tfp.distributions.MultivariateNormalFullCovariance(
                self.loc, covariance_matrix / self.mean_precision
            )
            lp += mvn.log_prob(mu)

        return lp

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


# Normal inverse Wishart prior
def _niw_from_stats(stats, counts):
    s_1, s_2, s_3 = stats
    dim = s_2.shape[-1]

    mean_precision = s_1
    # loc = lax.cond(mean_precision > 0,
    #             lambda x: s_2 / mean_precision,
    #             lambda x: np.zeros_like(s_2),
    #             None)
    loc = np.einsum("...i,...->...i", s_2, 1 / mean_precision)
    scale = s_3 - np.einsum("...,...i,...j->...ij", mean_precision, loc, loc)
    df = counts - dim - 2
    return NormalInverseWishart(loc, mean_precision, df, scale)


def _niw_pseudo_obs_and_counts(niw):
    pseudo_obs = (
        niw.mean_precision,
        niw.mean_precision * niw.loc,
        niw.scale
        + niw.mean_precision * np.einsum("...i,...j->...ij", niw.loc, niw.loc),
    )

    pseudo_counts = niw.df + niw.dim + 1
    return pseudo_obs, pseudo_counts


# Multivariate normal likelihood
def _mvn_from_params(params):
    loc, covariance = params
    scale_tril = np.linalg.cholesky(covariance)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)


def _mvn_suff_stats(data):
    return (np.ones(data.shape[:-1]), data, np.einsum("...i,...j->...ij", data, data))


# ss = tuple(s.sum(axis=0) for s in _mvn_suff_stats(data))
# niw = _niw_from_stats(ss, len(data))
# niw.mode()
