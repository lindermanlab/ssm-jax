"""
Useful plotting utility functions.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]

    Args:
        colors (list): List of color values (RGB or RGBA tuples).
        nsteps (int, optional): Number of steps in the gradient. Defaults to 256.
        bounds ([type], optional): [description]. Defaults to None.

    Returns:
        cmap: The gradient colormap.
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0,1,ncolors)


    reds = []
    greens = []
    blues = []
    alphas = []
    for b,c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap


def plot_dynamics_2d(dynamics_matrix,
                     bias_vector,
                     mins=(-40,-40),
                     maxs=(40,40),
                     npts=20,
                     axis=None,
                     **kwargs):
    """Utility to visualize the dynamics for a 2 dimensional dynamical system.

    Args:
        dynamics_matrix: 2x2 numpy array. "A" matrix for the system.
        bias_vector: "b" vector for the system. Has size (2,).
        mins: Tuple of minimums for the quiver plot.
        maxs: Tuple of maximums for the quiver plot.
        npts: Number of arrows to show.
        axis: Axis to use for plotting. Defaults to None, and returns a new axis.
        kwargs: keyword args passed to plt.quiver.

    Returns:
        q: quiver object returned by pyplot
    """
    assert dynamics_matrix.shape == (2, 2), "Must pass a 2 x 2 dynamics matrix to visualize."
    assert len(bias_vector) == 2, "Bias vector must have length 2."

    x_grid, y_grid = np.meshgrid(np.linspace(mins[0], maxs[0], npts), np.linspace(mins[1], maxs[1], npts))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((npts**2,0))))
    dx = xy_grid.dot(dynamics_matrix.T) + bias_vector - xy_grid

    if axis is not None:
        q = axis.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    else:
        q = plt.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)

    plt.gca().set_aspect(1.0)
    return q


def plot_single_sweep(particles, true_states, tag='', preprocessed=False, fig=None, _obs=None):
    """

    Stock code for plotting the results of an SMC sweep.

    Args:
        particles:
            N x T x D ndarray of particles to plot. -- OR -- if `preprocessed == True`, then this should be equal to a
            tuple ((TxD), (TxD), (TxD)) representing the median, lower and upper quantiles of the particles.  This is
            useful for when closed-form densities have been pre-computed.

        true_states:
            T x D ndarray of true latent states.

        tag:
            String to attach to the title of the plot.

        preprocessed:
            Bool indicating whether `particles` is an ndarray of particles, or, a preprocessed tuple of summary stats.\

        fig:
            plt.Figure object.  Allows a figure to be provided, which will be cleared and the sweep will be plotted.

    Returns:
        plt.Figure object on which the sweep was plotted.

    """
    # Define the standard plotting colours.
    color_names = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple"
    ]

    # Label just the first instance.
    gen_label = lambda _k, _s: _s if _k == 0 else None

    # Pull out the summary stats if they aren't preprocessed.
    if not preprocessed:
        single_sweep_median = np.median(particles, axis=0)
        single_sweep_lsd = np.quantile(particles, 0.17, axis=0)
        single_sweep_usd = np.quantile(particles, 0.83, axis=0)
    else:
        single_sweep_median = particles[0]
        single_sweep_lsd = particles[1]
        single_sweep_usd = particles[2]

    ts = np.arange(len(true_states))

    # Generate a new figure or clean the old figure.
    if fig is not None:
        plt.figure(fig.number)
        plt.clf()
    else:
        fig = plt.figure(figsize=(14, 6))

    # Plot the data.
    for _i, _c in zip(range(single_sweep_median.shape[1]), color_names):
        plt.plot(ts, single_sweep_median[:, _i], c=_c, label=gen_label(_i, 'Predicted'))
        plt.fill_between(ts, single_sweep_lsd[:, _i], single_sweep_usd[:, _i], color=_c, alpha=0.1)
        plt.plot(ts, true_states[:, _i], c=_c, linestyle='--', label=gen_label(_i, 'True'))

    # Finalize and draw the plot.
    plt.title(tag)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.pause(0.0001)

    return fig
