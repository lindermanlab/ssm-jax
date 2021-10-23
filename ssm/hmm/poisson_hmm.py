import jax.numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class, tree_map
from tensorflow_probability.substrates import jax as tfp

import ssm.distributions
from ssm.hmm.core import HMM

@register_pytree_node_class
class PoissonHMM(HMM):
    def __init__(self, num_states: int,
                 initial_distribution: tfp.distributions.Categorical,
                 transition_distribution: tfp.distributions.Categorical,
                 emission_distribution: tfp.distributions.Independent,
                 initial_distribution_prior: tfp.distributions.Dirichlet=None,
                 transition_distribution_prior: tfp.distributions.Dirichlet=None,
                 emission_distribution_prior: tfp.distributions.Gamma=None
                 ):

        """An HMM with Gaussian emissions.

        Args:
            num_states (int): Number of discrete latent states.
            initial_distribution (tfp.distributions.Categorical): The distribution over the initial state.
            transition_distribution (tfp.distributions.Categorical): The transition distribution.
            TODO
        """

        super(PoissonHMM, self).__init__(num_states,
                                         initial_distribution=initial_distribution,
                                         transition_distribution=transition_distribution,
                                         emission_distribution=emission_distribution,
                                         initial_distribution_prior=initial_distribution_prior,
                                         transition_distribution_prior=transition_distribution_prior,
                                         emission_distribution_prior=emission_distribution_prior)


    def _m_step_emission_distribution(self, dataset, posteriors):
        """If we have the right posterior, we can perform an exact update here.
        """
        # import pdb; pdb.set_trace()
        _compute_conditionals = \
            lambda weights: ssm.distributions.compute_conditional(
                "IndependentPoisson",
                dataset,
                weights=weights,
                prior=self._emission_distribution_prior)

        tmp = _compute_conditionals(posteriors.expected_states[:, :, 0])
        conditional = vmap(_compute_conditionals, in_axes=-1)(posteriors.expected_states)
        self._emission_distribution = \
            tfp.distributions.Independent(
                tfp.distributions.Poisson(conditional.mode()),
                reinterpreted_batch_ndims=1)
