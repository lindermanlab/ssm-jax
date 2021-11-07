import jax.numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp
import ssm.distributions as ssmd
tfd = tfp.distributions


class Emissions:
    """
    Base class of emission distribution of an HMM

    ..  math::
        p_t(x_t \mid z_t, u_t)

    where u_t are optional covariates.
    """
    def __init__(self, num_states: int) -> None:
        self._num_states = num_states

    @property
    def num_states(self):
        return self._num_states

    def distribution(self, state, covariates=None):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        raise NotImplementedError

    def log_probs(self, data):
        """
        Compute log p(x_t | z_t=k) for all t and k.
        """
        inds = np.arange(self.num_states)
        return vmap(lambda k: self.distribution(k).log_prob(data))(inds).T

    def m_step(self, dataset, posteriors):
        # TODO: implement generic m-step
        raise NotImplementedError


class ExponentialFamilyEmissions(Emissions):
    _emissions_distribution_class = None

    def __init__(self,
                 num_states: int,
                 emissions_distribution: ssmd.ExponentialFamilyDistribution=None,
                 emissions_distribution_prior: ssmd.ConjugatePrior=None) -> None:
        """Exponential Family Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (np.ndarray, optional): state-dependent emission means. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """

        super(ExponentialFamilyEmissions, self).__init__(num_states)
        self._distribution = emissions_distribution
        self._prior = emissions_distribution_prior

    def tree_flatten(self):
        children = (self._distribution, self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        distribution, prior = children
        return cls(aux_data,
                   emissions_distribution=distribution,
                   emissions_distribution_prior=prior)

    @property
    def emissions_dim(self):
        return self._distribution.event_shape

    def distribution(self, state: int, covariates: np.ndarray=None) -> ssmd.MultivariateNormalTriL:
        """Get the distribution at the provided state.

        Args:
            state (int): discrete state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions distribution (tfd.MultivariateNormalLinearOperator):
                emissions distribution at given state
        """
        return self._distribution[state]

    def m_step(self, dataset, posteriors):
        """Update the emissions distribution in-place using an M-step.

        Operates over a batch of data (posterior must have the same batch dim).

        Args:
            dataset (np.ndarray): the observed dataset
            posteriors ([type]): the HMM posteriors
        """
        conditional = self._emissions_distribution_class.compute_conditional(
            dataset, weights=posteriors.expected_states, prior=self._prior)
        self._distribution = self._emissions_distribution_class.from_params(
            conditional.mode())


@register_pytree_node_class
class BernoulliEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.IndependentBernoulli

    def __init__(self,
                 num_states: int,
                 probs: np.ndarray=None,
                 emissions_distribution: ssmd.MultivariateNormalTriL=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Gaussian Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            probs (np.ndarray, optional): state-dependent emission probabilities. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """

        assert probs is not None or emissions_distribution is not None

        if probs is not None:
            emissions_distribution = ssmd.IndependentBernoulli(probs=probs)

        if emissions_distribution_prior is None:
            emissions_distribution_prior = ssmd.Beta(1.1, 1.1)

        super(BernoulliEmissions, self).__init__(num_states,
                                                 emissions_distribution,
                                                 emissions_distribution_prior)


@register_pytree_node_class
class GaussianEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.MultivariateNormalTriL

    def __init__(self,
                 num_states: int,
                 means: np.ndarray=None,
                 covariances: np.ndarray=None,
                 emissions_distribution: ssmd.MultivariateNormalTriL=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        """Gaussian Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (np.ndarray, optional): state-dependent emission means. Defaults to None.
            covariances (np.ndarray, optional): state-dependent emission covariances. Defaults to None.
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """

        assert (means is not None and covariances is not None) \
            or emissions_distribution is not None

        if means is not None and covariances is not None:
            emissions_distribution = ssmd.MultivariateNormalTriL(means, covariances)

        super(GaussianEmissions, self).__init__(num_states,
                                                emissions_distribution,
                                                emissions_distribution_prior)


@register_pytree_node_class
class PoissonEmissions(ExponentialFamilyEmissions):
    _emissions_distribution_class = ssmd.IndependentPoisson

    def __init__(self,
                 num_states: int,
                 rates: np.ndarray=None,
                 emissions_distribution: ssmd.IndependentPoisson=None,
                 emissions_distribution_prior: ssmd.Gamma=None) -> None:
        """Poisson Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            rates (np.ndarray, optional): state-dependent Poisson rates. Defaults to None.
            emissions_distribution (tfd.Distribution, optional): pre-initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (tfd.Gamma, optional): pre-initialized emissions distribution prior.
                Defaults to None.
        """
        assert rates is not None or emissions_distribution is not None
        if rates is not None:
            emissions_distribution = ssmd.IndependentPoisson(rates)

        super(PoissonEmissions, self).__init__(num_states,
                                               emissions_distribution,
                                               emissions_distribution_prior)
