import matplotlib.pyplot as plt
import jax.numpy as np
import jax.random as jr
from ssm.lds.models import GaussianLDS
from ssm.deep_lds.posterior import DKFPosterior, LDSSVAEPosterior
from ssm.nn_util import inv_softplus, softplus, \
    BiRNNMeanParams, GaussianNetworkFullMeanParams

def lds_from_params(lds_params):
    eps = 1e-4
    m1, Q1, A, b, Q, dec_params = lds_params
    dec_mean_params = dec_params["params"]["head_mean_fn"]["Dense_0"]
    dec_cov_params = dec_params["params"]["head_log_var_fn"]
    C, d = dec_mean_params["kernel"].T, dec_mean_params["bias"]
    emission_cov = softplus(dec_cov_params["bias"]) + eps

    num_emission_dims = d.shape[0]
    num_latent_dims = b.shape[0]

    emission_scale_tril = emission_cov ** 0.5 * np.eye(num_emission_dims)
    dynamics_scale_tril = np.linalg.cholesky(Q)
    initial_state_scale_tril = np.linalg.cholesky(Q1)

    lds = GaussianLDS(num_latent_dims=num_latent_dims,
                    num_emission_dims=num_emission_dims,
                    initial_state_mean=m1,
                    initial_state_scale_tril=dynamics_scale_tril,
                    dynamics_weights=A,
                    dynamics_bias=b,
                    dynamics_scale_tril=dynamics_scale_tril,
                    emission_weights=C,
                    emission_bias=d,
                    emission_scale_tril=emission_scale_tril)
    return lds

# Returns a random projection matrix from ND to 2D
def random_projection(seed, N):
    key1, key2 = jr.split(seed, 2)
    v1 = jr.normal(key1, (N,))
    v2 = jr.normal(key2, (N,))

    v1 /= np.linalg.norm(v1)
    v2 -= v1 * np.dot(v1, v2)
    v2 /= np.linalg.norm(v2)

    return np.stack([v1, v2])

def get_gaussian_draw_params(mu, Sigma, proj_seed=None):
    if (mu.shape[0] > 2):
        P = random_projection(proj_seed, mu.shape[0])
        mu = P @ mu
        Sigma = P @ Sigma @ P.T
    angles = np.hstack([np.arange(0, 2*np.pi, 0.01), 0])
    circle = np.vstack([np.sin(angles), np.cos(angles)])
    ellipse = np.dot(2 * np.linalg.cholesky(Sigma), circle)
    return (mu[0], mu[1]), (ellipse[0, :] + mu[0], ellipse[1, :] + mu[1])

def plot_gaussian_2D(mu, Sigma, proj_seed=None, ax=None, **kwargs):
    """
    Helper function to plot 2D Gaussian contours
    """
    (px, py), (exs, eys) = get_gaussian_draw_params(mu, Sigma, proj_seed)

    ax = plt.gca() if ax is None else ax
    point = ax.plot(px, py, marker='D', **kwargs)
    line, = ax.plot(exs, eys, **kwargs)
    return (point, line)

# TODO: We should put this in the deep_lds class somehow?
def deep_lds_e_step(data, rec_params, model, rec_net,
                    posterior_class, **params):
    posterior = posterior_class.initialize(model, data, **params)
    if params["inference_method"] == "planet":
        rec_params, post_params = rec_params
        potentials = rec_net.apply(rec_params, data)
        posterior = posterior.update(model, data, potentials)
        # Temporarily boost the number of samples for the deep autoreg posterior
        num_samples = posterior_class.NUM_SAMPLES
        posterior_class.NUM_SAMPLES = 100
        posterior = posterior_class.update_params(posterior, post_params)
        posterior_class.NUM_SAMPLES = num_samples
    else:
        potentials = rec_net.apply(rec_params, data)
        posterior = posterior.update(model, data, potentials)
    return posterior

def get_marginals_and_targets(data, indices, lds, model, rec_net,
                          rec_params_history, model_params_history=None,
                          posterior_class=DKFPosterior,
                          **params):
    trials, times = indices
    num_pts = trials.shape[0]
    # Get the targets
    if model_params_history is None:
        posterior = lds.e_step(data)
        tgt_mus = [posterior.mean()[trials, times]]
        tgt_Sigmas = [posterior.covariance()[trials, times]]
    else:
        tgt_mus = []
        tgt_Sigmas = []
        for p in model_params_history:
            posterior = lds_from_params(p).e_step(data)
            tgt_mus.append(posterior.mean()[trials, times])
            tgt_Sigmas.append(posterior.covariance()[trials, times])

    mus = []
    Sigmas = []
    for p in rec_params_history:
        posterior = deep_lds_e_step(data, p, model, rec_net, 
            posterior_class, **params)
        mus.append(posterior.mean()[trials, times])
        Sigmas.append(posterior.covariance()[trials, times])
    return mus, Sigmas, tgt_mus, tgt_Sigmas