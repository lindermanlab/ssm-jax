import jax.numpy as np
import jax.random as jr
from jax import vmap
from jax.tree_util import tree_leaves, register_pytree_node_class

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
               metadata=None):
        from ssm.slds.models import GaussianSLDS
        if isinstance(slds, GaussianSLDS):
            continuous_posterior = \
                StructuredMeanFieldSLDSPosterior._cavi_update_continuous_posterior(
                    self.discrete_posterior, slds, data, covariates, metadata)
        else:
            raise NotImplementedError

        discrete_posterior = StructuredMeanFieldSLDSPosterior._cavi_update_discrete_posterior(
            continuous_posterior, slds, data, covariates, metadata)
        return StructuredMeanFieldSLDSPosterior(
            discrete_posterior, continuous_posterior)

    @staticmethod
    def _cavi_update_continuous_posterior(discrete_posterior,
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

        def _update_single(data, q_z):
            Ez = q_z.expected_states

            # diagonal blocks of precision matrix
            J_diag = np.einsum('tk,kij->tij', Ez, CTRiCs)
            # J_diag = J_diag.at[0].add(np.einsum('k, kij->ij', Ez[0], Q0is))
            J_diag = J_diag.at[0].add(Q0is)
            J_diag = J_diag.at[:-1].add(np.einsum('tk,kij->tij', Ez[1:], ATQiAs))
            J_diag = J_diag.at[1:].add(np.einsum('tk,kij->tij', Ez[1:], Qis))

            # lower diagonal blocks of precision matrix
            J_lower_diag = np.einsum('tk,kij->tij', Ez[1:], -QiAs)

            # linear potential
            h = np.einsum('tk, tkn, kni->ti', Ez, data[:, None, :] - ds, RiCs)
            # h = h.at[0].add(np.einsum('k,ki->i', Ez[0], Q0im1s))
            h = h.at[0].add(Q0im0s)
            h = h.at[:-1].add(-np.einsum('tk,ki->ti', Ez[1:], ATQibs))
            h = h.at[1:].add(np.einsum('tk,ki->ti', Ez[1:], Qibs))
            return LDSPosterior.infer(J_diag, J_lower_diag, h)

        return vmap(_update_single)(data, discrete_posterior)

    @staticmethod
    def _cavi_update_discrete_posterior(continuous_posterior,
                                        slds,
                                        data,
                                        covariates=None,
                                        metadata=None):

        def _update_single(q_x, data, covariates, metadata):
            # Shorthand names for parameters
            log_initial_states_probs = slds._discrete_initial_condition.log_initial_probs(
                data, covariates=covariates, metadata=metadata)
            log_transition_matrices = slds._transitions.log_transition_matrices(
                data, covariates=covariates, metadata=metadata)

            # Compute the expected emissions and dynamics log probs under the continuous posterior
            Ex = q_x.expected_states
            ExxT = q_x.expected_states_squared
            ExnxT = q_x.expected_states_next_states

            if covariates is not None:
                # TODO: extend expectations with covariates
                raise NotImplementedError

            expected_log_likelihoods = vmap(
                lambda yi, Exi, ExixiT: \
                    slds._emissions._distribution.expected_log_prob(
                        yi, Exi, np.outer(yi, yi), np.outer(yi, Exi), ExixiT))(
                            data, Ex, ExxT
                        )

            expected_log_likelihoods = expected_log_likelihoods.at[1:, :].add(
                vmap(slds._dynamics._distribution.expected_log_prob)(
                    Ex[1:], Ex[:-1], ExxT[1:], ExnxT, ExxT[:-1]))

            return StationaryHMMPosterior.infer(
                log_initial_states_probs, expected_log_likelihoods, log_transition_matrices
            )

        return vmap(_update_single)(continuous_posterior, data, covariates, metadata)

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