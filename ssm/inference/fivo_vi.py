
import jax
import jax.random as jr
import jax.numpy as np


def compute_single_elbo(_tilt_vmapped, _state_single, _obs_single):
    """

    Args:
        _tilt_vmapped:
        _state_single:
        _obs_single:

    Returns:

    """

    # Compute the tilt value (in log space), remembering that the final state doesn't have a tilt.
    _r_log_val = _tilt_vmapped(_state_single[:-1], np.arange(len(_state_single) - 1), _obs_single)

    return _r_log_val


def compute_elbo(_rebuild_tilt, _tilt_params, _model, _state_batch, _obs_batch):
    """

    Args:
        _rebuild_tilt:
        _vi_opt:
        _model:
        _state_batch:
        _obs_batch:

    Returns:

    """

    # Reconstruct the tilt, but don't bind an observation to it yet.
    _tilt = _rebuild_tilt(_tilt_params, None, _model)

    # Build a tilt function that we can apply at each timestep.
    _tilt_vmapped = jax.vmap(_tilt, in_axes=(0, 0, None))

    # Build a tilt function that we can apply at each timestep.
    _compute_single_vmapped = jax.vmap(compute_single_elbo, in_axes=(None, 0, 0))

    # Compute the tilt value (in log space).
    _r_log_vals = _compute_single_vmapped(_tilt_vmapped, _state_batch, _obs_batch)

    # Compute the ELBO as the mean of the tilts.
    _negative_elbo = - np.mean(_r_log_vals)

    return _negative_elbo


def do_vi_tilt_update(key,
                      _env,
                      _param_vals,
                      _rebuild_model,
                      _rebuild_tilt,
                      _rebuild_encoder,
                      _state_buffer_raw,
                      _obs_buffer_raw,
                      _mask_buffer_raw,
                      _vi_opt,
                      _epochs=5,
                      _sgd_batch_size=16):
    """

    Args:
        key:
        _env:
        _param_vals:
        _rebuild_model:
        _rebuild_tilt:
        _rebuild_encoder:
        _state_buffer_raw:
        _obs_buffer_raw:
        _mask_buffer_raw:
        _vi_opt:
        _epochs:
        _sgd_batch_size:

    Returns:

    """
    print('[test_message]: Hello, im an uncompiled VI update.')
    assert _vi_opt is not None

    if _env.config.model == 'VRNN':
        raise NotImplementedError()

    # Reconstruct the model, inscribing the current parameter values.
    model = _rebuild_model(_param_vals[0])

    #
    encoder = _rebuild_encoder(_param_vals[3])
    if encoder is not None:
        raise NotImplementedError("Don't quite know how to handle encoders here yet.")

    # Construct the batch.
    state_buffer_shaped = np.concatenate(_state_buffer_raw)
    obs_buffer_shaped = np.repeat(np.expand_dims(np.concatenate(_obs_buffer_raw), 1), state_buffer_shaped.shape[1], axis=1)

    state_buffer = state_buffer_shaped.reshape((-1, *state_buffer_shaped.shape[2:]))
    obs_buffer = obs_buffer_shaped.reshape((-1, *obs_buffer_shaped.shape[2:]))

    # Build up the objective function.
    elbo_closed = lambda _p, _x, _y: compute_elbo(_rebuild_tilt, _p, model, _x, _y)
    elbo_val_and_grad = jax.value_and_grad(elbo_closed, argnums=0)

    vi_gradient_steps = 0
    expected_elbo = 0.0

    def _single_epoch(carry, _t):

        (__vi_opt, ) = carry

        state_batch = state_buffer[idxes_batch[_t]]
        obs_batch = obs_buffer[idxes_batch[_t]]

        elbo, grad = elbo_val_and_grad(__vi_opt.target, state_batch, obs_batch)

        __vi_opt = __vi_opt.apply_gradient(grad)

        return (__vi_opt, ), elbo

    # Loop over the epochs.
    for _epoch in range(_epochs):

        # Construct the batches.
        key, subkey = jr.split(key)
        idxes = jr.permutation(subkey, np.arange(len(obs_buffer)))

        if len(idxes) % _sgd_batch_size == 0:
            idxes_trimmed = idxes
        else:
            idxes_trimmed = idxes[0:-(len(idxes) % _sgd_batch_size)]
        idxes_batch = np.reshape(idxes_trimmed, (-1, _sgd_batch_size))

        (_vi_opt, ), elbos = jax.lax.scan(_single_epoch, (_vi_opt, ), (np.arange(len(idxes_batch))))

        vi_gradient_steps += len(idxes_batch)
        expected_elbo = np.mean(elbos)

    return _vi_opt, expected_elbo, vi_gradient_steps