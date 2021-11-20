import jax.numpy as np
import jax.scipy.special as spsp
from jax import lax, value_and_grad, vmap

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization

class SVAEPosterior(tfp.distributions.Distribution):
    def __init__(self):
        pass

    @classmethod
    def infer(cls, data):
        # What information should be passed in here?
        # I feel like this class shouldn't really do all of the work
        # And we should really just separate the CRF part from the NN part
        pass

