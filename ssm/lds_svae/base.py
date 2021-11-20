import jax.numpy as np
import jax.random as jr
from jax import lax, vmap

from ssm.base import SSM

@register_pytree_node_class
class SVAE(SSM):
    def __init__(self):
        # Initial distribution
        # Dynamics
        # Generation network (emissions)
        # Ideally, we want to pass in the recognition network in here as well
        pass
    
    @property
    def emissions_shape(self):
        pass
    
    def initial_distribution(self, covariates=None, metadata=None):
        pass

    def dynamics_distribution(self, state, covariates=None, metadata=None):
        pass

    def emissions_distribution(self, state, covariates=None, metadata=None):
        pass

    # Auto-batching?
    def e_step(self, data, covariates=None, metadata=None):
        return SVAEPosterior.infer(data)

    def m_step(self, dataset, posteriors):
        pass

    def marginal_likelihood(self, data, posterior):
        pass