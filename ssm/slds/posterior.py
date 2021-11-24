import jax.numpy as np
import jax.random as jr
from jax import lax, vmap, hessian, jacfwd, jacrev
from jax.tree_util import tree_leaves, register_pytree_node_class
import jax.scipy.optimize

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from ssm.hmm.posterior import StationaryHMMPosterior
from ssm.distributions import MultivariateNormalBlockTridiag as LDSPosterior
from ssm.utils import one_hot


@register_pytree_node_class
class StructuredMeanFieldSLDSPosterior():
    r"""A tfp.Distribution-like object representing a posterior distribution
    on discrete and continuous latent states of an SLDS.

    ..math:
        q(z, x) = q(z) q(x) = \mathrm{HMMPosterior}(z) \times \mathrm{LDSPosterior}(x)

    Trying to implement this as a subclass of ``tfd.JointDistributionNamed`` presented
    more problems than it was worth.
    """
    def __init__(self,
                 discrete_posterior: StationaryHMMPosterior=None,
                 continuous_posterior: LDSPosterior=None) -> None:

        self._discrete_posterior = discrete_posterior
        self._continuous_posterior = continuous_posterior

    @property
    def discrete_posterior(self):
        return self._discrete_posterior

    @property
    def continuous_posterior(self):
        return self._continuous_posterior

    def sample(self, sample_shape=(), seed=None):
        key1, key2 = jr.split(seed, 2)
        return dict(
            discrete=self.discrete_posterior.sample(sample_shape, seed=key1),
            continuous=self.continuous_posterior.sample(sample_shape, seed=key2))

    def log_prob(self, value):
        return self.discrete_posterior.log_prob(value["discrete"]) + \
            self.continuous_posterior.log_prob(value["continuous"])

    @classmethod
    def initialize(cls, slds, data, covariates=None, metadata=None, scale=10.0):
        """
        Initialize a structured mean field posterior given a model and data.
        """
        num_states = slds.num_states
        latent_dim = slds.latent_dim
        num_batches = tree_leaves(data)[0].shape[0]
        num_timesteps = tree_leaves(data)[0].shape[1]

        # TODO: use self.emissions_shape
        flat_dataset = data.reshape(-1, data.shape[-1])

        # Run PCA on the flattened data
        from sklearn.decomposition import PCA
        pca = PCA(latent_dim)
        flat_continuous_states = pca.fit_transform(flat_dataset)
        continuous_states = flat_continuous_states.reshape(num_batches, num_timesteps, latent_dim)
        emissions_matrix = pca.components_.T
        emissions_variance = pca.noise_variance_
        precision_diag_blocks = (emissions_matrix.T / emissions_variance) @  emissions_matrix
        precision_diag_blocks = np.tile(precision_diag_blocks, (num_timesteps, 1, 1))
        precision_lower_diag_blocks = np.zeros((num_timesteps - 1, latent_dim, latent_dim))
        continuous_posterior = \
            vmap(LDSPosterior.infer_from_precision_and_mean, in_axes=(None, None, 0))(
                precision_diag_blocks, precision_lower_diag_blocks, continuous_states)

        # Run KMeans on the PCs
        from sklearn.cluster import KMeans
        km = KMeans(num_states)
        assignments = km.fit_predict(flat_continuous_states).reshape(num_batches, num_timesteps)
        log_likelihoods = scale * one_hot(assignments, num_states)
        log_initial_state_probs = np.zeros(num_states)
        log_transition_matrix = np.zeros((num_states, num_states))
        discrete_posterior = \
            vmap(StationaryHMMPosterior.infer, in_axes=(None, 0, None))(
                log_initial_state_probs, log_likelihoods, log_transition_matrix)

        return cls(discrete_posterior, continuous_posterior)

    def update(self,
               slds,
               data,
               covariates=None,
               metadata=None,
               key=None):
        from ssm.slds.models import GaussianSLDS
        if isinstance(slds, GaussianSLDS):
            self._cavi_update_continuous_posterior(slds, data, covariates, metadata)
            self._cavi_update_discrete_posterior(slds, data, covariates, metadata)
        else:
            self._laplace_update_continuous_posterior(slds, data, covariates, metadata)
            self._monte_carlo_update_discrete_posterior(key, slds, data, covariates, metadata)

        # Update discrete posterior q(z)
        return self

    def _cavi_update_continuous_posterior(self,
                                          gaussian_slds,
                                          data,
                                          covariates=None,
                                          metadata=None):
        """
        Compute the exact posterior by extracting the natural parameters
        of the LDS, namely the block tridiagonal precision matrix (J) and
        the linear coefficient (h).
        """
        # Shorthand names for parameters
        m0s = gaussian_slds.initial_mean
        Q0s = gaussian_slds.initial_covariance
        As = gaussian_slds.dynamics_weights
        bs = gaussian_slds.dynamics_biases
        Qs = gaussian_slds.dynamics_covariances
        Cs = gaussian_slds.emissions_weights
        ds = gaussian_slds.emissions_biases
        Rs = gaussian_slds.emissions_covariances

        # Precompute natural parameters for each state
        transpose = lambda x: np.swapaxes(x, -1, -2)
        Q0is = np.linalg.inv(Q0s)
        Q0im0s = np.dot(Q0is, m0s)
        Qis = np.linalg.inv(Qs)
        QiAs = Qis @ As
        Qibs = (Qis @ bs[:, :, None])[:, :, 0]
        ATQiAs = transpose(As) @ QiAs
        ATQibs = (transpose(As) @ Qibs[:, :, None])[:, :, 0]
        RiCs = np.linalg.solve(Rs, Cs)
        CTRiCs = transpose(Cs) @ RiCs

        def _update_single(y, Ez):
            # diagonal blocks of precision matrix
            J_diag = np.einsum('tk,kij->tij', Ez, CTRiCs)
            # J_diag = J_diag.at[0].add(np.einsum('k, kij->ij', Ez[0], Q0is))
            J_diag = J_diag.at[0].add(Q0is)
            J_diag = J_diag.at[:-1].add(np.einsum('tk,kij->tij', Ez[1:], ATQiAs))
            J_diag = J_diag.at[1:].add(np.einsum('tk,kij->tij', Ez[1:], Qis))

            # lower diagonal blocks of precision matrix
            J_lower_diag = np.einsum('tk,kij->tij', Ez[1:], -QiAs)

            # linear potential
            h = np.einsum('tk, tkn, kni->ti', Ez, y[:, None, :] - ds, RiCs)
            # h = h.at[0].add(np.einsum('k,ki->i', Ez[0], Q0im1s))
            h = h.at[0].add(Q0im0s)
            h = h.at[:-1].add(-np.einsum('tk,ki->ti', Ez[1:], ATQibs))
            h = h.at[1:].add(np.einsum('tk,ki->ti', Ez[1:], Qibs))
            return LDSPosterior.infer(J_diag, J_lower_diag, h)

        self._continuous_posterior = vmap(_update_single)(
            data, self.discrete_posterior.expected_states)

    def _laplace_update_continuous_posterior(self,
                                             slds,
                                             data,
                                             covariates=None,
                                             metadata=None,
                                             num_lbfgs_iters=50):

        def _expected_log_prob(x, y, Ez):
            # initial distribution log prob
            lp = slds._continuous_initial_condition._distribution.log_prob(x[0])
            # expected dynamics log prob
            f = lambda xtm1, xt, Ezt: np.dot(Ezt, slds._dynamics._distribution.log_prob(xt, covariates=xtm1))
            lp += vmap(f)(x[:-1], x[1:], Ez[1:]).sum()
            # expected emissions log prob
            f = lambda xt, yt, Ezt: np.dot(Ezt, slds._emissions._distribution.log_prob(yt, covariates=xt))
            lp += vmap(f)(x, y, Ez).sum()
            return lp

        def _compute_mean(x0, y, Ez):
            scale = x0.size
            dim = x0.shape[-1]

            objective = lambda x_flat: -1 * _expected_log_prob(x_flat.reshape(-1, dim), y, Ez) / scale
            optimize_results = jax.scipy.optimize.minimize(
                objective,
                x0.ravel(),
                method="l-bfgs-experimental-do-not-rely-on-this",
                options=dict(maxiter=num_lbfgs_iters))
            return optimize_results.x.reshape(-1, dim)

        def _compute_precision_blocks(x, y, Ez):

            # initial distribution hessian
            J_init = -1 * hessian(slds._continuous_initial_condition._distribution.log_prob)(x[0])
            # dynamics hessian
            f = lambda xtm1, xt, Ezt: np.dot(Ezt, slds._dynamics._distribution.log_prob(xt, covariates=xtm1))
            J_11 = -1 * vmap(hessian(f, argnums=0))(x[:-1], x[1:], Ez[1:])
            J_22 = -1 * vmap(hessian(f, argnums=1))(x[:-1], x[1:], Ez[1:])
            J_21 = -1 * vmap(jacfwd(jacrev(f, argnums=1), argnums=0))(x[:-1], x[1:], Ez[1:])
            # emissions hessian
            f = lambda xt, yt, Ezt: np.dot(Ezt, slds._emissions._distribution.log_prob(yt, covariates=xt))
            J_obs = -1 * vmap(hessian(f, argnums=0))(x, y, Ez)

            # combine into diagonal and lower diagonal blocks
            J_diag = J_obs
            J_diag = J_diag.at[0].add(J_init)
            J_diag = J_diag.at[:-1].add(J_11)
            J_diag = J_diag.at[1:].add(J_22)
            J_lower_diag = J_21
            return J_diag, J_lower_diag

        def _update_single(args):
            x0, y, Ez = args
            x = _compute_mean(x0, y, Ez)
            J_diag, J_lower_diag = _compute_precision_blocks(x, y, Ez)
            return LDSPosterior.infer_from_precision_and_mean(J_diag, J_lower_diag, x)

        self._continuous_posterior = lax.map(_update_single,
                                             (self.continuous_posterior.expected_states,
                                              data,
                                              self.discrete_posterior.expected_states))

    def _cavi_update_discrete_posterior(self,
                                        gaussian_slds,
                                        data,
                                        covariates=None,
                                        metadata=None):

        def _update_single(q_x, y, u, m):
            # Shorthand names for parameters
            log_initial_states_probs = gaussian_slds._discrete_initial_condition.log_initial_probs(
                y, covariates=u, metadata=m)
            log_transition_matrices = gaussian_slds._transitions.log_transition_matrices(
                y, covariates=u, metadata=m)

            # Compute the expected emissions and dynamics log probs under the continuous posterior
            Ex = q_x.expected_states
            ExxT = q_x.expected_states_squared
            ExnxT = q_x.expected_states_next_states

            if u is not None:
                # TODO: extend expectations with covariates
                raise NotImplementedError

            expected_log_likelihoods = vmap(
                lambda yi, Exi, ExixiT: \
                    gaussian_slds._emissions._distribution.expected_log_prob(
                        yi, Exi, np.outer(yi, yi), np.outer(yi, Exi), ExixiT))(
                            y, Ex, ExxT
                        )

            expected_log_likelihoods = expected_log_likelihoods.at[1:, :].add(
                vmap(gaussian_slds._dynamics._distribution.expected_log_prob)(
                    Ex[1:], Ex[:-1], ExxT[1:], ExnxT, ExxT[:-1]))

            return StationaryHMMPosterior.infer(
                log_initial_states_probs, expected_log_likelihoods, log_transition_matrices
            )

        self._discrete_posterior = vmap(_update_single)(
            self.continuous_posterior, data, covariates, metadata)

    def _monte_carlo_update_discrete_posterior(self,
                                               key,
                                               slds,
                                               data,
                                               covariates=None,
                                               metadata=None):

        def _update_single(k, q_x, y, u, m):
            # Shorthand names for parameters
            log_initial_states_probs = slds._discrete_initial_condition.log_initial_probs(
                y, covariates=u, metadata=m)
            log_transition_matrices = slds._transitions.log_transition_matrices(
                y, covariates=u, metadata=m)

            # Compute the expected emissions and dynamics log probs under the continuous posterior
            Ex = q_x.expected_states
            ExxT = q_x.expected_states_squared
            ExnxT = q_x.expected_states_next_states

            if u is not None:
                # TODO: extend expectations with covariates
                raise NotImplementedError

            # approximate the emissions log likelihood with a sample of x ~ q(x)
            x = q_x.sample(seed=k)
            expected_log_likelihoods = vmap(
                lambda yi, xi: slds._emissions._distribution.log_prob(yi, covariates=xi))(y, x)

            # Compute the dynamics log prob exactly
            expected_log_likelihoods = expected_log_likelihoods.at[1:, :].add(
                vmap(slds._dynamics._distribution.expected_log_prob)(
                    Ex[1:], Ex[:-1], ExxT[1:], ExnxT, ExxT[:-1]))

            return StationaryHMMPosterior.infer(
                log_initial_states_probs, expected_log_likelihoods, log_transition_matrices
            )

        self._discrete_posterior = vmap(_update_single)(
            jr.split(key, len(data)), self.continuous_posterior, data, covariates, metadata)


    def tree_flatten(self):
        children = (self.discrete_posterior, self.continuous_posterior)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __getitem__(self, slices):
        return StructuredMeanFieldSLDSPosterior(
            self._discrete_posterior[slices],
            self._continuous_posterior[slices]
        )

    def __str__(self):
        return f"<ssm.slds.posterior.{type(self).__name__} "\
               f"discrete={self.discrete_posterior} " \
               f"continuous={self.continuous_posterior}"
