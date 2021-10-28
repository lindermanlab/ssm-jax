"""

"""

import jax.numpy as np
import jax.random as jr
import jax.scipy.special as spsp
from jax import lax, value_and_grad, vmap
from jax.tree_util import register_pytree_node_class

from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.python.internal import reparameterization



def forward_pass_stationary(initial_potentials,
                            unary_potentials,
                            pairwise_potentials):
    """
    Compute the marginal likelihood (i.e. log normalizer) under
    a chain structured graphical model with discrete variables.

    The normalizer is computed via the forward message passing recursion,

    .. math::
        \log \\alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \\alpha_{t,i} +
                                \phi_{t, i} +
                                \phi_{i, j} \}

    where,

    .. math::
        \\alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).

    Args:
        initial_potentials: Log likelihood of first state in a ``(K,)``
            array where ``K`` is the number of states.

        unary_potentials: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.

        pairwise_potentials: Log transition matrix or matrices. The shape
            must either be ``(K, K)`` where ``K`` is the number of states.
            The ``[i,j]`` entry gives the log probability of transitioning
            from state ``z_t = i`` to state ``z_{t+1} = j``.

    Returns:
        The log probability of the sequence ``x[:T]``, summing over the
        discrete states.
    """
    def marginalize(alpha, ll):
        alpha = spsp.logsumexp(alpha + ll + pairwise_potentials.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(marginalize,
                               initial_potentials,
                               unary_potentials[:-1])

    # Include the initial potentials to get log Pr(z_t | x_{1:t-1})
    # for all time steps. These are the "filtered potentials".
    filtered_potentials = np.row_stack([initial_potentials, alphas])

    # Account for the last timestep when computing marginal lkhd
    return spsp.logsumexp(alpha_T + unary_potentials[-1]), filtered_potentials


def forward_pass_nonstationary(initial_potentials,
                               unary_potentials,
                               pairwise_potentials):
    """
    Compute the marginal likelihood (i.e. log normalizer) under
    a chain structured graphical model with discrete variables.

    The normalizer is computed via the forward message passing recursion,

    .. math::
        \log \\alpha_{t+1,j} =
            \log \sum_i \exp \{ \log \\alpha_{t,i} +
                                \phi_{t, i} +
                                \phi_{t, i, j} \}

    where,

    .. math::
        \\alpha_{t+1} \propto p(z_{t+1} | x_{1:t}).

    Args:
        initial_potentials: Log likelihood of first state in a ``(K,)``
            array where ``K`` is the number of states.

        unary_potentials: Log likelihoods combined in a shape ``(T, K)``
            array where ``T`` is the length of the sequence and ``K`` is
            the number of states.  The ``[t, i]`` entry specifies the log
            likelihood of observation ``x[t]`` given state ``z[t] = i``.

        pairwise_potentials: Log transition matrix or matrices. The shape
            must either be ``(T-1, K, K)`` where ``T`` is the length of the
            sequence and ``K`` is the number of states. The ``[t,i,j]`` entry
            gives the log probability of transitioning from state ``z_t = i`` to
            state ``z_{t+1} = j`` for ``t=1,...,T-1``.

    Returns:
        The log probability of the sequence ``x[:T]``, summing over the
        discrete states.
    """
    T, K = unary_potentials.shape
    assert initial_potentials.shape == (K,)
    assert pairwise_potentials.shape == (T-1, K, K)

    def _step(alpha, prms):
        unary, pair = prms
        alpha = spsp.logsumexp(alpha + unary + pair.T, axis=1)
        return alpha, alpha

    alpha_T, alphas = lax.scan(
        _step,
        initial_potentials,
        (unary_potentials[:-1], pairwise_potentials),
    )

    # Include the initial potentials to get log Pr(z_t | x_{1:t-1})
    # for all time steps. These are the "filtered potentials".
    filtered_potentials = np.row_stack([initial_potentials, alphas])

    # Account for the unary potential at the last timestep
    return spsp.logsumexp(alpha_T + unary_potentials[-1]), filtered_potentials


class _DiscreteChain(tfp.distributions.Distribution):
    """
    TODO
    """
    def __init__(self,
                 initial_potentials,
                 unary_potentials,
                 pairwise_potentials,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="DiscreteChain",
             ) -> None:
        # self.check_shapes(initial_potentials, unary_potentials, pairwise_potentials)
        self._unary_potentials = unary_potentials
        self._initial_potentials = initial_potentials
        self._pairwise_potentials = pairwise_potentials

        # Keep track of whether or not we've run message passing
        self._ran_message_passing = False

        super(_DiscreteChain, self).__init__(
            dtype=int,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=dict(initial_potentials=self._initial_potentials,
                            unary_potentials=self._unary_potentials,
                            pairwise_potentials=self._pairwise_potentials),
            name=name,
        )

    def tree_flatten(self):
        children = (self._initial_potentials,
                    self._unary_potentials,
                    self._pairwise_potentials)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def num_timesteps(self):
        return self._unary_potentials.shape[-2]

    @property
    def num_states(self):
        return self._unary_potentials.shape[-1]

    def _message_passing(self):
        raise NotImplementedError

    @property
    def log_normalizer(self):
        if not self._ran_message_passing:
            self._message_passing()
        return self._log_normalizer

    def _mean(self):
        if not self._ran_message_passing:
            self._message_passing()
        return self._expected_states

    @property
    def expected_states(self):
        return self.mean()

    @property
    def expected_transitions(self):
        if not self._ran_message_passing:
            self._message_passing()
        return self._expected_transitions

    def _entropy(self):
        raise NotImplementedError


class StationaryDiscreteChain(_DiscreteChain):
    """
    TODO
    """
    def check_shapes(self,
                     initial_potentials,
                     unary_potentials,
                     pairwise_potentials):
        assert unary_potentials is not None and unary_potentials.ndim >= 2
        assert initial_potentials is not None and initial_potentials.ndim >= 1
        assert pairwise_potentials is not None and pairwise_potentials.ndim >= 2
        num_timesteps, num_states = unary_potentials.shape[-2:]

        assert initial_potentials.shape[-1] == num_states
        assert pairwise_potentials.shape[:-2] == unary_potentials.shape[:-2]
        assert pairwise_potentials.shape[-2:] == (num_states, num_states)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            initial_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            unary_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            pairwise_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
        )

    def _message_passing(self):
        # Run message passing code to get the log normalizer, the filtered potentials,
        # and the expected values of the discrete variables.
        def _message_passing_single(initial_potentials,
                                    unary_potentials,
                                    pairwise_potentials):
            f = value_and_grad(forward_pass_stationary, argnums=(1, 2), has_aux=True)
            (_log_normalizer, _alphas), (_expected_states, _expected_transitions) = \
                f(initial_potentials, unary_potentials, pairwise_potentials)
            return _log_normalizer, _alphas, _expected_states, _expected_transitions

        T, K = self.num_timesteps, self.num_states
        batch_shape = self.batch_shape
        initial_potentials = self._initial_potentials.reshape(-1, K)
        unary_potentials = self._unary_potentials.reshape(-1, T, K)
        pairwise_potentials = self._pairwise_potentials.reshape(-1, K, K)

        _log_normalizer, _alphas, _expected_states, _expected_transitions = \
            vmap(_message_passing_single)(initial_potentials,
                                          unary_potentials,
                                          pairwise_potentials)

        self._log_normalizer = _log_normalizer.reshape(batch_shape)
        self._alphas = _alphas.reshape(batch_shape + (T, K))
        self._expected_states = _expected_states.reshape(batch_shape + (T, K))
        self._expected_transitions = _expected_transitions.reshape(batch_shape + (K, K))
        self._ran_message_passing = True

    def _log_prob(self, data, **kwargs):
        T, K = self.num_timesteps, self.num_states
        batch_shape = self.batch_shape
        flat_data = data.reshape(-1, T)
        flat_unary_potentials = self._unary_potentials.reshape(-1, T, K)
        flat_pairwise_potentials = self._pairwise_potentials.reshape(-1, K, K)

        def _log_prob_single(z, unary, pair):
            lp = unary[np.arange(T), z].sum()
            lp += pair[z[:-1], z[1:]].sum()
            lp -= self.log_normalizer
            return lp

        lps = vmap(_log_prob_single)(flat_data,
                                     flat_unary_potentials,
                                     flat_pairwise_potentials)

        return lps.reshape(batch_shape)

    def _sample_n(self, n, seed=None):
        if not self.ran_message_passing:
            self.message_passing()

        # TODO make these parameters w/ lazy op?
        alphas = self._alphas

        raise NotImplementedError

        # def sample_single(seed, filtered_Js, filtered_hs, J_lower_diag):

        #     def _step(carry, inpt):
        #         x_next, seed = carry
        #         Jf, hf, L = inpt

        #         # Condition on the next observation
        #         Jc = Jf
        #         hc = hf - x_next @ L

        #         # Split the seed
        #         seed, this_seed = jr.split(seed)
        #         x = _sample_info_gaussian(this_seed, Jc, hc, sample_shape)
        #         return (x, seed), x

        #     # Initialize with sample of last state
        #     seed_T, seed = jr.split(seed)
        #     x_T = _sample_info_gaussian(seed_T, filtered_Js[-1], filtered_hs[-1], sample_shape)
        #     inputs = (filtered_Js[:-1][::-1], filtered_hs[:-1][::-1], J_lower_diag[::-1])
        #     _, x_rev = lax.scan(_step, (x_T, seed), inputs)
        #     return np.concatenate((x_rev[::-1], x_T[None, ...]), axis=0)

        # # batch mode
        # if filtered_Js.ndim == 4:
        #     samples = vmap(sample_single)(seed, filtered_Js, filtered_hs, J_lower_diag)

        # # non-batch mode
        # else:
        #     samples = sample_single(seed, filtered_Js, filtered_hs, J_lower_diag)
        # return samples

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ez = self.expected_states
        Ezzp1 = self.expected_transitions
        h = self._unary_potentials
        J = self._pairwise_potentials

        entropy = -np.sum(h * Ez, axis=(-2, -1))
        entropy += -np.sum(J * Ezzp1, axis=(-2, -1))
        entropy += self.log_normalizer
        return entropy

class NonstationaryDiscreteChain(_DiscreteChain):
    """
    TODO
    """
    def check_shapes(self,
                     initial_potentials,
                     unary_potentials,
                     pairwise_potentials):

        assert unary_potentials is not None and unary_potentials.ndim >= 2
        assert initial_potentials is not None and initial_potentials.ndim >= 1
        assert pairwise_potentials is not None and pairwise_potentials.ndim >= 3

        num_timesteps, num_states = unary_potentials.shape[-2:]
        assert initial_potentials.shape[-1] == num_states
        assert pairwise_potentials.shape[:-3] == unary_potentials.shape[:-2]
        assert pairwise_potentials.shape[-3:] == (num_timesteps - 1, num_states, num_states)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            initial_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=1),
            unary_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=2),
            pairwise_potentials=tfp.internal.parameter_properties.ParameterProperties(event_ndims=3),
        )

    def _message_passing(self):
        # Run message passing code to get the log normalizer, the filtered potentials,
        # and the expected values of the discrete variables.
        def _message_passing_single(initial_potentials,
                                    unary_potentials,
                                    pairwise_potentials):
            f = value_and_grad(forward_pass_nonstationary, argnums=(1, 2), has_aux=True)
            (_log_normalizer, _alphas), (_expected_states, _expected_transitions) = \
                f(initial_potentials, unary_potentials, pairwise_potentials)
            return _log_normalizer, _alphas, _expected_states, _expected_transitions

        T, K = self.num_timesteps, self.num_states
        batch_shape = self.batch_shape
        initial_potentials = self._initial_potentials.reshape(-1, K)
        unary_potentials = self._unary_potentials.reshape(-1, T, K)
        pairwise_potentials = self._pairwise_potentials.reshape(-1, T-1, K, K)

        _log_normalizer, _alphas, _expected_states, _expected_transitions = \
            vmap(_message_passing_single)(initial_potentials,
                                          unary_potentials,
                                          pairwise_potentials)

        self._log_normalizer = _log_normalizer.reshape(batch_shape)
        self._alphas = _alphas.reshape(batch_shape + (T, K))
        self._expected_states = _expected_states.reshape(batch_shape + (T, K))
        self._expected_transitions = _expected_transitions.reshape(batch_shape + (T-1, K, K))
        self._ran_message_passing = True

    def _log_prob(self, data, **kwargs):
        T, K = self.num_timesteps, self.num_states
        batch_shape = self.batch_shape
        flat_data = data.reshape(-1, T)
        flat_unary_potentials = self._unary_potentials.reshape(-1, T, K)
        flat_pairwise_potentials = self._pairwise_potentials.reshape(-1, T-1, K, K)

        def _log_prob_single(z, unary, pair):
            lp = unary[np.arange(T), z].sum()
            lp += pair[np.arange(T-1), z[:-1], z[1:]].sum()
            lp -= self.log_normalizer
            return lp

        lps = vmap(_log_prob_single)(flat_data,
                                     flat_unary_potentials,
                                     flat_pairwise_potentials)

        return lps.reshape(batch_shape)

    def _sample_n(self, n, seed=None):
        if not self.ran_message_passing:
            self.message_passing()

        # TODO make these parameters w/ lazy op?
        alphas = self._alphas

        raise NotImplementedError

        # def sample_single(seed, filtered_Js, filtered_hs, J_lower_diag):

        #     def _step(carry, inpt):
        #         x_next, seed = carry
        #         Jf, hf, L = inpt

        #         # Condition on the next observation
        #         Jc = Jf
        #         hc = hf - x_next @ L

        #         # Split the seed
        #         seed, this_seed = jr.split(seed)
        #         x = _sample_info_gaussian(this_seed, Jc, hc, sample_shape)
        #         return (x, seed), x

        #     # Initialize with sample of last state
        #     seed_T, seed = jr.split(seed)
        #     x_T = _sample_info_gaussian(seed_T, filtered_Js[-1], filtered_hs[-1], sample_shape)
        #     inputs = (filtered_Js[:-1][::-1], filtered_hs[:-1][::-1], J_lower_diag[::-1])
        #     _, x_rev = lax.scan(_step, (x_T, seed), inputs)
        #     return np.concatenate((x_rev[::-1], x_T[None, ...]), axis=0)

        # # batch mode
        # if filtered_Js.ndim == 4:
        #     samples = vmap(sample_single)(seed, filtered_Js, filtered_hs, J_lower_diag)

        # # non-batch mode
        # else:
        #     samples = sample_single(seed, filtered_Js, filtered_hs, J_lower_diag)
        # return samples

    def _entropy(self):
        """
        Compute the entropy

            H[X] = -E[\log p(x)]
                 = -E[-1/2 x^T J x + x^T h - log Z(J, h)]
                 = 1/2 <J, E[x x^T] - <h, E[x]> + log Z(J, h)
        """
        Ez = self.expected_states
        Ezzp1 = self.expected_transitions
        h = self._unary_potentials
        J = self._pairwise_potentials

        entropy = -np.sum(h * Ez, axis=(-2, -1))
        entropy += -np.sum(J * Ezzp1, axis=(-3, -2, -1))
        entropy += self.log_normalizer
        return entropy
