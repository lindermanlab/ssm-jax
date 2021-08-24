from jax import vmap
import jax.numpy as np
import jax.scipy.special as spsp
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp


from ssm.models.base import SSM


@register_pytree_node_class
class HMM(SSM):

    def __init__(self, num_states,
                 initial_distribution,
                 transition_distribution,
                 emission_distribution):
        """ TODO
        """
        self.num_states = num_states

        # Set up the initial state distribution and prior
        assert isinstance(initial_distribution, tfp.distributions.Categorical)
        self._initial_distribution = initial_distribution

        assert isinstance(transition_distribution, tfp.distributions.Categorical)
        self._transition_distribution = transition_distribution

        assert isinstance(transition_distribution, tfp.distributions.Distribution)
        self._emissions_distribution = emission_distribution

    def tree_flatten(self):
        children = (self._initial_distribution,
                    self._transition_distribution,
                    self._emissions_distribution)
        aux_data = (self.num_states,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_states, = aux_data
        initial_distribution, transition_distribution, emission_distribution = children

        # Construct a new HMM
        return cls(num_states,
                   initial_distribution=initial_distribution,
                   transition_distribution=transition_distribution,
                   emission_distribution=emission_distribution)

    def initial_distribution(self):
        return self._initial_distribution

    def dynamics_distribution(self, state):
        return self._transition_distribution[state]

    def emissions_distribution(self, state):
        return self._emissions_distribution[state]

    @property
    def initial_state_probs(self):
        return self._initial_distribution.probs_parameter()

    @property
    def transition_matrix(self):
        return self._transition_distribution.probs_parameter()

    def natural_parameters(self, data):
        log_initial_state_distn = self._initial_distribution.logits_parameter()
        log_transition_matrix = self._transition_distribution.logits_parameter()
        log_transition_matrix -= spsp.logsumexp(log_transition_matrix, axis=1, keepdims=True)
        log_likelihoods = vmap(lambda k:
                               vmap(lambda x: self._emissions_distribution[k].log_prob(x))(data)
                               )(np.arange(self.num_states)).T

        return log_initial_state_distn, log_transition_matrix, log_likelihoods


class HMMConjugatePrior(object):
    def log_prob(self, hmm):
        raise NotImplementedError

    @property
    def initial_prior(self):
        return self._initial_prior

    @property
    def transition_prior(self):
        return self._transition_prior

    @property
    def emissions_prior(self):
        return self._emissions_prior


def _make_standard_hmm(num_states, initial_state_probs=None,
                       initial_state_logits=None,
                       transition_matrix=None,
                       transition_logits=None):
    # Set up the initial state distribution and prior
    if initial_state_logits is None:
        if initial_state_probs is None:
            initial_state_logits = np.zeros(num_states)
        else:
            initial_state_logits = np.log(initial_state_probs)

    initial_dist = tfp.distributions.Categorical(logits=initial_state_logits)

    # Set up the transition matrix and prior
    if transition_logits is None:
        if transition_matrix is None:
            transition_logits = np.zeros((num_states, num_states))
        else:
            transition_logits = np.log(transition_matrix)

    transition_dist = tfp.distributions.Categorical(logits=transition_logits)

    return initial_dist, transition_dist
