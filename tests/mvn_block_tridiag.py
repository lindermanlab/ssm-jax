import jax.numpy as np
import jax.random as jr

def make_big_Jh(key, T=10, D=2):
    k1, k2, k3 = jr.split(key, 3)
    J_diag_sqrt = jr.normal(k1, shape=(T, D, D))
    J_diag = np.einsum('tij,tji->tij', J_diag_sqrt, J_diag_sqrt)
    J_lower_diag = jr.normal(k2, shape=(T-1, D, D))
    h = jr.normal(k3, shape=(T, D))

    big_J = np.zeros((T * D, T * D))
    for t in range(T):
        tslc = slice(t * D, (t+1) * D)
        big_J = big_J.at[tslc, tslc].add(J_diag[t])

    for t in range(T-1):
        tslc = slice(t * D, (t+1) * D)
        tp1slc = slice((t+1) * D, (t+2) * D)
        big_J = big_J.at[tp1slc, tslc].add(J_lower_diag[t])
        big_J = big_J.at[tslc, tp1slc].add(J_lower_diag[t].T)

    big_h = h.ravel()

    return J_diag, J_lower_diag, h, big_J, big_h


def test_mean_to_h(key, T=10, D=2):
    """
    This conversion is performed in the MultivariateNormalBlockTridiag constructor.
    """
    k1, k2 = jr.split(key, 2)
    mean = jr.normal(k1, shape=(T, D))
    J_diag, J_lower_diag, _, big_J, _ = make_big_Jh(k2, T=T, D=D)

    # Compute the linear potential naively
    linear_potential_comp = (big_J @ mean.ravel()).reshape((T, D))

    # Compute with block tridiagonal math
    linear_potential = np.einsum('tij,tj->ti', J_diag, mean)

    # linear_potential[:-1] += np.einsum('tji,tj->ti', precision_lower_diag_blocks, mean[1:])
    linear_potential = linear_potential.at[:-1].add(
        np.einsum('tji,tj->ti', J_lower_diag, mean[1:]))

    # linear_potential[1:] += np.einsum('tij,tj->ti', precision_lower_diag_blocks, mean[:-1])
    linear_potential = linear_potential.at[1:].add(
        np.einsum('tij,tj->ti', J_lower_diag, mean[:-1]))

    assert np.allclose(linear_potential, linear_potential_comp)
    print("success")


if __name__ == "__main__":
    key = jr.PRNGKey(0)
    test_mean_to_h(key)
