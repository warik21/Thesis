import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_distribution(X, p, q, title):
    """
    Plot the source and target distributions
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.plot(X, p, 'b-', label='Source dist: p')
    plt.plot(X, q, 'r-', label='Target dist: q')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_marginals(X, p, q, title):
    """
    Plot the marginals of the transport map
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.plot(X, p, 'b-.', label='Source dist: p')
    plt.plot(X, q, 'r-.', label='Target dist: q')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_signed_marginals(X, p, q, Transport_plan, title):
    """
    Plot the marginals of the transport map
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.plot(X, p, 'b-.', label='Source dist: p (signed)')
    plt.plot(X, q, 'r-.', label='Target dist: q (signed)')
    n_p = len(p)
    n_q = len(q)
    dx = np.ones(n_p) / n_p
    dy = np.ones(n_q) / n_q
    plt.plot(X, (Transport_plan.T @ dx) * np.sign(q), 'k-', label='Final source dist (q): Transport_plan.T dx (signed)')
    plt.plot(X, (Transport_plan @ dy) * np.sign(p), 'g-', label='Final target dist (p): Transport_plan dy (signed)')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_transport_map(p, q, Transport_plan, title):
    """
    Plot the transport map
    :param p: source distribution
    :param q: target distribution
    :param X: linear space in which the source distribution is defined
    :param Y: linear space in which the target distribution is defined
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the transport map
    """
    plt.figure(figsize=(8, 8))
    plot1D_mat(p, q, Transport_plan, title)
    plt.show()


def plot_transport_map_with_marginals(p, q, Transport_plan, title):
    """
    Plot the transport map
    :param q: target distribution
    :param p: source distribution
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the transport map
    """
    plt.figure(figsize=(8, 8))
    plot1D_mat(p, q, Transport_plan, title)
    plt.show()


def plot1D_mat(a, b, M, title=''):
    """ Plot matrix M  with the source and target 1D distribution

    Creates a subplot with the source distribution a on the left and
    target distribution b on the tot. The matrix M is shown in between.


    Parameters
    ----------
    a : np.array, shape (na,)
        Source distribution
    b : np.array, shape (nb,)
        Target distribution
    M : np.array, shape (na,nb)
        Matrix to plot
    title: string, optional (default='')
    """
    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = plt.subplot(gs[0, 1:])
    plt.bar(xb, b, label='Target distribution')
    plt.yticks(())
    plt.title(title)

    # because of barh syntax, a and xa should be in reverse order.
    ax2 = plt.subplot(gs[1:, 0])
    plt.barh(xa, a, label='Source distribution')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(())

    ax3 = plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    img = plt.imshow(M, interpolation='nearest')
    plt.axis('off')

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(img, cax=cax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.2)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    """ Plot matrix M  in 2D with  lines using p values

    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.


    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    xt : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        paameters given to the plot functions (default color is black if
        nothing given)
    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                         alpha=G[i, j] / mx, **kwargs)


def plot1D_mat_bar(a, b, M, title=''):
    """ Plot matrix M  with the source and target 1D distribution

    Creates a subplot with the source distribution a on the left and
    target distribution b on the tot. The matrix M is shown in between.


    Parameters
    ----------
    a : np.array, shape (na,)
        Source distribution
    b : np.array, shape (nb,)
        Target distribution
    M : np.array, shape (na,nb)
        Matrix to plot
    title: string, optional (default='')
    """

    na, nb = M.shape

    gs = gridspec.GridSpec(3, 3)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax1 = plt.subplot(gs[0, 1:])
    plt.bar(xb, b, label='Target distribution')
    plt.yticks(())
    plt.title(title)

    # because of barh syntax, a and xa should be in reverse order.
    ax2 = plt.subplot(gs[1:, 0])
    plt.barh(xa, a, label='Source distribution')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(())

    plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    plt.imshow(M, interpolation='nearest', cmap='gray')
    plt.axis('off')

    plt.xlim((0, nb))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.2)
