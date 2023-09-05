from utils.utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_marginals(X, p, q, title, first_distribution_title='Source dist: p',
                   second_distribution_title='Target dist: q'):
    """
    Plot the marginals of the transport map

    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param title: title of the plot
    :param first_distribution_title: the legend label of the first distribution
    :param second_distribution_title: the legend label of the second distribution
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.bar(X, p, alpha=0.5, label=first_distribution_title, width=1 / len(p))
    plt.bar(X, q, alpha=0.5, label=second_distribution_title, width=1 / len(q))
    plt.title(title),
    plt.legend()
    plt.show()


def plot_transport_map_with_marginals(a, b, M, title=''):
    """ Plot matrix M  with the source and target 1D distribution

    Creates a subplot with the source distribution 'a' on the left and
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
    img = plt.imshow(M, interpolation='nearest', cmap='gray')
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
        parameters given to the plot functions (default color is black if
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
