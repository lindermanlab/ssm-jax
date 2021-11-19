import jax.numpy as np
import jax.scipy.special as spsp
from jax import lax, value_and_grad, vmap

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization

class SVAEPosterior(tfp.distributions.Distribution):
    def __init__(self):
        pass

    def infer(self, )