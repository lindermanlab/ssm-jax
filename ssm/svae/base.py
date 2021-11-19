import jax.numpy as np
import jax.random as jr
from jax import lax, vmap

from ssm.base import SSM

@register_pytree_node_class
class SVAE(SSM):
    def __init__(self):
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

    def e_step(self, data):
        pass
    
    def m_step(self, dataset, posteriors):
        pass

    def marginal_likelihood(self, data, posterior):
        pass