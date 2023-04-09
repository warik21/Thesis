import numpy as np
import matplotlib.pyplot as plt
import ot
from matplotlib import gridspec
from utils.utils import full_scalingAlg

def div0(x, y):
    """
    Special x/y with convention x/0=0
    """
    return np.divide(x, y, out=np.zeros_like(x), where=y != 0)


def mul0(x, y):
    """
    Return x*y with the convention 0*Inf = 0
    """
    return np.multiply(x, y, out=np.zeros_like(x), where=~np.isinf(y))


def proxdiv(F, s, u, eps, params):
    """
    Proxdiv operator of the divergence function F

    F: String description of the target function
    s:
    u:
    eps: epsilon parameter of the entropic regularization
    params: List of parameters of the corresponding F

    F = 'KL' --> params[0] = lda ; params[1] = p
    """

    if F == 'KL':
        lda = params[0]
        p = params[1]
        # return div0(p,s)**(lda/(lda+eps)) * np.exp(-u/(lda+eps))
        return div0(p, s * np.exp(u / lda)) ** (lda / (lda + eps))

    if F == 'TV':
        lda = params[0]
        p = params[1]
        term1 = np.exp((- u + lda) / eps)
        # print(u.shape)
        # print(term1.shape)
        # print(p.shape)
        # print(s.shape)
        # print((np.exp((-lda - u)/eps)).shape)
        # print((div0(p,s)).shape)
        term2 = np.maximum(np.exp((-lda - u) / eps), div0(p, s))
        return np.minimum(term1, term2)

    else:
        print('Not recognized function.')
        return


def fdiv(F, x, p, dx, params):
    """
    Divergence function F

    F: String description of the divergence function
    x: Test distribution (KL_F(x|p))
    p: Reference distribution (KL_F(x|p))
    dx: discretization vector
    params: List of parameters of the corresponding F

    F = 'KL' --> params[0] = lda
    """

    if F == 'KL':
        lda = params[0]
        return lda * np.sum(mul0(dx, (mul0(x, np.log(div0(x, p))) - x + p)))

    elif F == 'TV':
        lda = params[0]
        return lda * np.sum(mul0(dx, abs(x - p)))

    else:
        print('Not recognized function.')
        return


def fdiv_c(F, x, p, dx, params):
    """
    Convex conjugate of the divergence function F

    F: String description of the divergence function
    x: Test distribution (KL_F(x|p))
    p: Reference distribution (KL_F(x|p))
    dx: discretization vector
    params: List of parameters of the corresponding F

    F = 'KL' --> params[0] = lda
    """

    if F == 'KL':
        lda = params[0]
        return lda * np.sum(mul0(p * dx, np.exp(x / lda) - 1))

    elif F == 'TV':
        lda = params[0]
        return lda * np.sum(mul0(dx, np.minimum(p, np.maximum(-p, mul0(p, x / lda)))))

    else:
        print('Not recognized function.')
        return


def simple_scalingAlg(C, Fun, p, q, eps, dx, dy, n_max):
    """
    Simple implementation for solving Unbalanced OT problems

    C: Cost matrix
    Fun: List defining the function and its lambda parameter. e.i. Fun = ['KL', 0.01]
    p: Source distribution
    q: target dstribution
    eps: epsilon parameter
    dx: discretizaiton vector in x / np.shape(dx) = (nJ,1)
    dy: discretization vector in y / np.shape(dy) = (nJ,1)
    n_max: Max number of iterations
    """

    # Init
    nI = C.shape[0]
    nJ = C.shape[1]
    a_t = np.ones([nI, 1])
    b_t = np.ones([nJ, 1])
    pdgap = np.zeros([n_max, 1])
    F = Fun[0]
    lda = Fun[1]  # define lambda parameter value

    K_t = np.exp(C / (-eps))

    # Main Loop
    for it in range(n_max):  # -> Use for and cutting condition
        params = [lda, p]
        a_t = proxdiv(F, (K_t @ (b_t * dy)), 0., eps, params)

        params = [lda, q]
        b_t = proxdiv(F, (K_t.T @ (a_t * dx)), 0., eps, params)

        # Gap calculation
        R = (np.tile(a_t, nJ) * K_t) * np.tile(b_t, nI).T
        param_p = [lda]
        primal = fdiv(F, np.dot(R, dy), p, dx, param_p) + fdiv(F, np.dot(R.T, dx), q, dy, param_p) + \
                 eps / (nI * nJ) * np.sum(mul0(R, np.log(div0(R, K_t))) - R + K_t)
        dual = - fdiv_c(F, -eps * np.log(a_t), p, dx, param_p) - fdiv_c(F, -eps * np.log(b_t), q, dy, param_p) - \
               eps / (nI * nJ) * np.sum(R - K_t)
        pdgap[it] = primal - dual

    R = (np.tile(a_t, nJ) * K_t) * np.tile(b_t, nI).T

    return R, pdgap, a_t, b_t


def make_1D_gauss(n, m, s):
    """
    Return a 1D histogram for a gaussian distribution (n bins, mean m and std s)
    Parameters
    ----------
    n : int
        number of bins in the histogram
    m : float
        mean value of the gaussian distribution
    s : float
        standard deviaton of the gaussian distribution
    Returns
    -------
    h : np.array (n,)
          1D histogram for a gaussian distribution
    """
    x = np.arange(n, dtype=np.float64)
    h = np.exp(-(x - m) ** 2 / (2 * s ** 2))
    h = h / h.sum()
    return np.reshape(h, (len(h), 1))


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
    plt.plot(xb, b, 'r', label='Target distribution')
    plt.yticks(())
    plt.title(title)

    ax2 = plt.subplot(gs[1:, 0])
    plt.plot(a, xa, 'b', label='Source distribution')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks(())

    plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
    plt.imshow(M, interpolation='nearest')
    plt.axis('off')

    plt.xlim((0, nb))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.2)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    """ Plot matrix M  in 2D with  lines using alpha values

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


def full_scalingAlg_pot(source, target, costs, reg_param):
    """
    Implementation for solving ot using sinkhorn, including log-domain stabilization
    Also works on Unbalanced data

    source(np.ndarray): The source distribution, p
    target(np.ndarray): The target distribution, q
    costs(np.ndarray): The cost matrix
    reg_param(float): Regularization parameter, epsilon in the literature
    """
    K_t : np.ndarray = np.exp(costs / (-reg_param))
    Transport_cost, logs = ot.sinkhorn(source, target, costs, reg=reg_param, log=True)
    u : np.ndarray = logs['u'].flatten()
    v : np.ndarray = logs['v'].flatten()
    Transport_plan : np.ndarray = np.diag(u) @ K_t @ np.diag(v)

    return Transport_plan, u, v

def signed_GWD(C, Fun, p, q, eps_vec, dx, dy, n_max, verb=True, eval_rate=10):
    p_pos = np.zeros(p.shape)
    p_neg = np.zeros(p.shape)
    q_pos = np.zeros(q.shape)
    q_neg = np.zeros(q.shape)

    sign_p = np.sign(p)
    sign_q = np.sign(q)

    p_pos[sign_p > 0] = p[sign_p > 0]
    p_neg[sign_p < 0] = -p[sign_p < 0]
    q_pos[sign_q > 0] = q[sign_q > 0]
    q_neg[sign_q < 0] = -q[sign_q < 0]

    p_tilde = p_pos + q_neg
    q_tilde = q_pos + p_neg

    return full_scalingAlg(C, Fun, p_tilde, q_tilde, eps_vec, dx, dy, n_max, verb, eval_rate)
