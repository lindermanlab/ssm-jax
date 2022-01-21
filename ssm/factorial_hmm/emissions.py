from jax import vmap
import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from flax.core.frozen_dict import freeze, FrozenDict

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import ssm.distributions as ssmd
from ssm.hmm.emissions import Emissions


class FactorialEmissions(Emissions):

    def __init__(self, num_states: tuple):
        super().__init__(num_states)
        self._num_groups = len(num_states)

    @property
    def num_groups(self):
        return self._num_groups


@register_pytree_node_class
class NormalFactorialEmissions(FactorialEmissions):
    """
    
    """
    def __init__(self, num_states: tuple,
                 means: (tuple or list)=None,
                 log_scale: float=0.0,
                 emissions_distribution: tfd.Normal=None,
                 emissions_distribution_prior: ssmd.NormalInverseWishart=None) -> None:
        r"""Normal Emissions for a Factorial HMM.
        
        The emission mean is a sum of means associated with each group.
        
        .. math::
            x_t | \{z_{tj} \}_{j=1}^J ~ N(\sum_j m_{z_{tj}}, \sigma^2)

        Can be initialized by specifying parameters or by passing in a pre-initialized
        ``emissions_distribution`` object.

        Args:
            num_states (tuple): number of discrete latent states per group
            means (tuple or list, optional): state-dependent and group-dependent emission means. Defaults to None.
            variance (np.ndarray, optional): emission variance shared by all states
            emissions_distribution (ssmd.MultivariateNormalTriL, optional): initialized emissions distribution.
                Defaults to None.
            emissions_distribution_prior (ssmd.NormalInverseWishart, optional): initialized emissions distribution prior.
                Defaults to None.
        """
        super().__init__(num_states)
        if means is not None:
            big_means = np.zeros(num_states)
            for k, m in enumerate(means):
                m_expanded = np.expand_dims(m, axis=range(1, self.num_groups))
                big_means += np.swapaxes(m_expanded, 0, k)
            scale = np.exp(log_scale.astype(np.float32)) # ensure not weakly typed else 2x jit
            emissions_distribution = tfd.Normal(big_means, scale)

        self._means = means
        self._distribution = emissions_distribution
        self._prior = emissions_distribution_prior

    def tree_flatten(self):
        # children = (self._distribution, self._prior)
        # aux_data = self.num_states
        children = (self._means, np.log(self._distribution.scale), self._prior)
        aux_data = self.num_states
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        means, log_scale, prior = children
        return cls(aux_data,
                   means=means, log_scale=log_scale,
                   emissions_distribution_prior=prior)

    @property
    def emissions_shape(self):
        return self._distribution.event_shape
    
    @property
    def _parameters(self):
        return freeze(dict(distribution=self._distribution))
        
    @_parameters.setter
    def _parameters(self, params):
        self._distribution = params["distribution"]
        
    @property
    def _hyperparameters(self):
        return freeze(dict(prior=self._prior))
    
    @_hyperparameters.setter
    def _hyperparameters(self, hyperparams):
        self._prior = hyperparams["prior"]

    def distribution(self, state, covariates=None, metadata=None):
        """
        Return the conditional distribution of emission x_t
        given state z_t and (optionally) covariates u_t.
        """
        return self._distribution[state]

    def log_likelihoods(self, data, covariates=None, metadata=None):
        """
        Compute log p(x_t | z_t=(k_1, ..., k_J)) for all t and (k_1,...,k_J).
        """
        return vmap(self._distribution.log_prob)(data)
