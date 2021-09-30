from re import A
import jax.numpy as np
import jax.random as jr

from ssm.models.slds import GaussianSLDS
from ssm.utils import random_rotation

def random_slds(key, K=3, D=2, N=10):
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 10)
    pi0_logits = jr.normal(k0, shape=(K,))
    P_logits = jr.normal(k1, shape=(K, K))
    m0s = jr.normal(k2, shape=(K, D))
    Q0_sqrts = jr.normal(k3, shape=(K, D, D))
    As = np.array([random_rotation(k4i, D) for k4i in jr.split(k4, K)])
    bs = jr.normal(k5, shape=(K, D))
    Q_sqrts = jr.normal(k6, shape=(K, D, D))
    Cs = jr.normal(k7, shape=(K, N, D))
    ds = jr.normal(k8, shape=(K, N))
    R_sqrts = jr.normal(k9, shape=(K, N, N))

    return GaussianSLDS(
        initial_state_logits=pi0_logits,
        transition_logits=P_logits,
        initial_means=m0s,
        initial_scale_trils=Q0_sqrts,
        dynamics_matrices=As,
        dynamics_biases=bs,
        dynamics_scale_trils=Q_sqrts,
        emissions_matrices=Cs,
        emissions_biases=ds,
        emissions_scale_trils=R_sqrts
    )

def test_basic(key, K=3, D=2, N=10, T=100):
    k1, k2 = jr.split(key, 2)
    slds = random_slds(k1, K=K, D=D, N=N)
    states, data = slds.sample(k2, num_steps=T)
    assert isinstance(states, dict)
    assert 'z' in states and 'x' in states
    assert states['z'].shape == (T,) and states['z'].dtype == np.int32
    assert states['x'].shape == (T, D) and states['x'].dtype == np.float32
    assert data.shape == (T, N) and data.dtype == np.float32


if __name__ == "__main__":
    test_basic(jr.PRNGKey(0))
