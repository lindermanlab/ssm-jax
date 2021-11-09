import jax.numpy as np
from jax import vmap

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.hmm.emissions import Emissions

class FactorialEmissions(Emissions):
    pass


class NormalFactorialEmissions(FactorialEmissions):
    """
    x_t | \{z_{tg} \}_{g=1}^G ~ N(\sum_g m_{z_{tg}}, \sum_g \sigma^2_{z_{tg}})
    """
    def __init__(self, num_states: int,
                 group_means: (tuple or list)=None,
                 variance: float=1.0,
                 emissions_distribution: ssmd.MultivariateNormalTriL=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        super().__init__(num_states)
        """Normal Emissions for HMM.

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (int): number of discrete states
            means (tuple or list, optional): state-dependent and group-dependent emission means. Defaults to None.
            variance (np.ndarray, optional): emission variance shared by all states
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """
        if group_means is not None:
            means = np.einsum()
            emissions_distribution = tfd.Normal(means, variance)

        self._num_states = num_states
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

    def distribution(self, state):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        return self._distribution[state]

    def log_probs(self, data):
        """
        Compute log p(x_t | z_t=(k1, ..., kG)) for all t and (k1,...,kG).
        """
        return vmap(self._distribution.log_prob)(data)

    def m_step(self, dataset, posteriors):
        def _objective(params):
            group_means, log_variance = params
            dist = NormalFactorialEmissions(self.num_states, group_means, np.exp(log_variance))
            f = lambda data, expected_states: np.sum(dist.log_probs(data) * expected_states)
            lp = vmap(f)(dataset, posteriors.expected_states)
            return -lp / dataset.size

        # TODO: Optimize the objective with jax.scipy.optimize.minimize like in laplace_em




