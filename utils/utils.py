import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.Classes import TransportResults
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import ot
import jax.numpy as jnp
from scipy.special import logsumexp
import cvxpy as cp


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


def normalize_array(x: np.ndarray):
    """
    normalizes an array to have a sum of one
    x: np.ndarray: the array
    """
    x = x / x.sum()
    return x


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


def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


def solve_ott(x, y):
    geom = pointcloud.PointCloud(x, y, epsilon=1e-1)
    prob = linear_problem.LinearProblem(geom)

    solver = sinkhorn.Sinkhorn(threshold=1e-2, lse_mode=True, max_iterations=1000)
    out = solver(prob)

    f, g = out.f, out.g
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = jnp.where(out.converged, jnp.sum(f) + jnp.sum(g), jnp.nan)
    return f, g, reg_ot


def solve_ot(a, b, x, y, ep, threshold):
    _, log = ot.sinkhorn(a, b, ot.dist(x, y), ep,
                         stopThr=threshold, method="sinkhorn_stabilized", log=True, numItermax=1000)

    f, g = ep * log["logu"], ep * log["logv"]
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = (np.sum(f * a) + np.sum(g * b) if log["err"][-1] < threshold else np.nan)

    return f, g, reg_ot


def create_constraints(source, target):
    """
    This function takes two lists as input and creates a matrix variable and a set of constraints.

    Parameters:
    - source (list): A list of non-negative numbers representing the source distribution.
    - target (list): A list of non-negative numbers representing the target distribution.

    Returns:
    - T_matrix (cvxpy.Variable): A matrix variable with shape (len(source), len(target)) representing the transport plan.
    - cons (list): A list of cvxpy constraints.

    Constraints:
    - The sum of each column of T_matrix is equal to the corresponding element of target.
    - The sum of each row of T_matrix is equal to the corresponding element of source.
    - T_matrix is element-wise non-negative.
    """
    T_matrix = cp.Variable((len(source), len(target)), nonneg=True)

    # noinspection PyTypeChecker
    cons = [cp.sum(T_matrix, axis=0) == target,  # column sum should be what we move to the pixel the column represents
            cp.sum(T_matrix, axis=1) == source,  # row sum should be what we move from the pixel the row represents
            T_matrix >= 0]  # all elements of T_matrix should be non-negative

    return T_matrix, cons


def create_constraints_lifted(source, target):
    """
    This function takes two real measures as input and creates a matrix variable and a set of constraints. While
    considering a lifting parameter p which is used to normalize the program.
    """
    T_matrix = cp.Variable((len(source), len(target)), nonneg=True)
    # alpha = cp.Variable(nonneg=True)  # lifting parameter
    alpha = -min(min(source), min(target)) + 1  # lifting parameter

    cons = [cp.sum(T_matrix, axis=0) == target + alpha,
            # column sum should be what we move to the pixel the column represents
            cp.sum(T_matrix, axis=1) == source + alpha,
            # row sum should be what we move from the pixel the row represents
            cp.sum(T_matrix) <=
            T_matrix >= 0]  # all elements of the transport plan should be non-negative

    return T_matrix, alpha, cons


def solve_ot_dual(c, mu, nu):
    n, m = c.shape  # n and m are the dimensions of cost matrix c
    phi = cp.Variable(n)
    psi = cp.Variable(m)

    constraints = [phi[i] + psi[j] <= c[i, j] for i in range(n) for j in range(m)]

    objective = cp.Maximize(mu @ phi + nu @ psi)
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return phi.value, psi.value, problem.value


def create_constraints_signed(source, target):
    T_matrix_pos = cp.Variable((len(source), len(target)), nonneg=True)
    T_matrix_neg = cp.Variable((len(source), len(target)), nonneg=True)

    cons = [cp.sum(T_matrix_pos - T_matrix_neg, axis=0) == target,
            # column sum should be what we move to the pixel the column represents
            cp.sum(T_matrix_pos - T_matrix_neg, axis=1) == source,
            # row sum should be what we move from the pixel the row represents
            T_matrix_pos >= 0,  # all elements of both matrices should be non-negative
            T_matrix_neg >= 0]

    return T_matrix_pos, T_matrix_neg, cons


def create_T(source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray, transport_type: str):
    """
    This function takes a TransportResults object and updates it according to the transport type.

    """
    if transport_type == 'standard':
        p, constraints = create_constraints(source, target)
        obj = cp.Minimize(cp.sum(cp.multiply(p, cost_matrix)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return TransportResults(transported_mass=prob.value, transport_plan=p.value,
                                source_distribution=source, target_distribution=target)

    elif transport_type == 'lifted':
        T, alpha, constraints = create_constraints_lifted(source.flatten(), target.flatten())
        obj = cp.Minimize(cp.sum(cp.multiply(T, cost_matrix)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return TransportResults(transported_mass=prob.value, transport_plan=T.value, lift_parameter=alpha,
                                source_distribution=source, target_distribution=target)

    elif transport_type == 'signed':
        T_pos, T_neg, constraints = create_constraints_signed(source.flatten(), target.flatten())
        obj = cp.Minimize(cp.sum(cp.multiply(T_pos, cost_matrix)) + cp.sum(cp.multiply(T_neg, cost_matrix)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return TransportResults(transported_mass=prob.value, Pos_plan=T_pos.value, Neg_plan=T_neg.value,
                                transport_plan=T_pos.value - T_neg.value,
                                source_distribution=source, target_distribution=target)

    else:
        raise ValueError('Invalid transport type. Must be either "standard", "lifted" or "signed".')


def calc_transport_cvxpy(source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray,
                         transport_type: str = 'standard'):
    """
    This function takes two lists and a matrix as input and solves a linear transport problem.

    Parameters:
    - source (numpy.ndarray): A list of non-negative numbers representing the source distribution.
    - target (numpy.ndarray): A list of non-negative numbers representing the target distribution.
    - cost_matrix (numpy.ndarray): A matrix representing the transport cost from each source to each target.
    - transport_type (str): A string representing the type of transport problem to solve. Can be either 'standard', 'lifted' or 'signed'.

    Returns:
    - (float, numpy.ndarray): A tuple containing the optimal transport cost and the optimal transport plan.

    The linear transport problem being solved is:
    Minimize (sum of element-wise product of transport plan and cost matrix)
    Subject to constraints:
    - The sum of each column of transport plan is equal to the corresponding element of target.
    - The sum of each row of transport plan is equal to the corresponding element of source.
    - transport plan is element-wise non-negative.
    """
    T = create_T(source, target, cost_matrix, transport_type)

    return T.transport_plan, T.transported_mass


def calc_lifted_transport_cvxpy(source, target, cost_matrix):
    """
    This function takes two real measures and a cost matrix as input to solve a linear transport problem.

    Parameters:
    - source (list): A list of non-negative numbers representing the source distribution.
    - target (list): A list of non-negative numbers representing the target distribution.
    - cost_matrix (numpy.ndarray): A matrix representing the transport cost from each source to each target.

    Returns:
    - (float, numpy.ndarray): A tuple containing the optimal transport cost and the optimal transport plan.

    The linear transport problem being solved is:
    Minimize (sum of element-wise product of transport plan and cost matrix)
    Subject to constraints:
    - The sum of each column of transport plan is equal to the corresponding element of target.
    - The sum of each row of transport plan is equal to the corresponding element of source.
    - transport plan is element-wise non-negative.
    """
    T, alpha, constraints = create_constraints_lifted(source, target)

    obj = cp.Minimize(cp.sum(cp.multiply(T, cost_matrix)))

    prob = cp.Problem(obj, constraints)

    prob.solve()

    return prob.value, T.value


def full_scalingAlg(C, Fun, p, q, eps_vec, dx, dy, n_max, verb=False, eval_rate=10):
    """
    Implementation for solving Unbalanced OT problems that includes the log-domain stabilization

    C: Cost matrix
    Fun: List defining the function and its lambda parameter. e.i. Fun = ['KL', 0.01]
    p: Source distribution
    q: target dstribution
    eps_vec: epsilon parameter (If scalar, the same epsilons is used throughout the algorithm.
        If it is a vector, the epsilons are equally distributed along the iterations forcing an absorption
        at each epsilon change.)
    dx: discretizaiton vector in x / np.shape(dx) = (nJ,1)
    dy: discretization vector in y / np.shape(dy) = (nJ,1)
    n_max: Max number of iterations
    """

    # Initialization
    nI = C.shape[0]
    nJ = C.shape[1]
    a_t = np.ones([nI, 1])
    u_t = np.zeros([nI, 1])
    b_t = np.ones([nJ, 1])
    v_t = np.zeros([nJ, 1])
    F = Fun[0]
    lda = Fun[1]  # define lambda parameter value
    eps_ind = 0  # index for the chosen epsilons
    eval_rate = eval_rate
    n_evals = np.floor(n_max / eval_rate).astype(int)
    param_p = [lda]
    primals = np.zeros((n_evals, 1))
    duals = np.zeros((n_evals, 1))
    pdgaps = np.zeros((n_evals, 1))

    if np.isscalar(eps_vec):
        eps = eps_vec
        eps_tot = 1
    else:
        eps = eps_vec[eps_ind]  # select the first epsilon to use
        eps_tot = len(eps_vec)

    K_t = np.exp(C / (-eps))

    # Main Loop
    for it in range(n_max):  # -> Use for and cutting condition
        params = [lda, p]
        a_t = proxdiv(F, (K_t @ (b_t * dy)), u_t, eps, params)

        params = [lda, q]
        b_t = proxdiv(F, (K_t.T @ (a_t * dx)), v_t, eps, params)

        if verb:
            # Check the primal, dual and primal-dual gap
            if it % eval_rate == 0:
                it_eval = 0
                R = (np.tile(a_t, nJ) * K_t) * np.tile(b_t, nI).T  # Reconstruct map
                primal = fdiv(F, R @ dy, p, dx, param_p) + fdiv(F, R.T @ dx, q, dy, param_p) + \
                         eps / (nI * nJ) * np.sum(mul0(R, np.log(div0(R, K_t))) - R + K_t)
                dual = - fdiv_c(F, -eps * np.log(a_t), p, dx, param_p) - fdiv_c(F, -eps * np.log(b_t), q, dy, param_p) - \
                       eps / (nI * nJ) * np.sum(R - K_t)
                pdgap = primal - dual

                # print("primal = %f \n dual = %f \n pdgap = %f \n"%(primal,dual,pdgap))

                primals[it_eval] = primal
                duals[it_eval] = dual
                pdgaps[it_eval] = pdgap

                it_eval += 1

        # stabilizations
        # print('it/n_max = %f , (eps_ind+1)/len(eps_vec) = %f'%(it/n_max , (eps_ind+1)/len(eps_vec)))
        if np.max([abs(a_t), abs(b_t)]) > 1e50 or (it / n_max) > (eps_ind + 1) / eps_tot:  # or it == n_max-1:
            """
            primal = fdiv(F,Transport_plan@dy,p,dx,param_p) + fdiv(F,Transport_plan.T@dx,q,dy,param_p) + \
                eps/(nI*nJ) * np.sum( mul0(Transport_plan , np.log(div0(Transport_plan,K_t))) - Transport_plan + K_t )
            dual = - fdiv_c(F,-eps*np.log(a_t),p,dx,param_p) - fdiv_c(F,-eps*np.log(b_t),q,dy,param_p) -\
                eps/(nI*nJ) * np.sum(Transport_plan-K_t)
            pdgap = primal-dual
            """

            # absorb
            u_t = u_t + eps * np.log(a_t)
            v_t = v_t + eps * np.log(b_t)

            if (it / n_max) > (eps_ind + 1) / eps_tot:
                eps_ind += 1
                eps = eps_vec[eps_ind]

            # update K
            K_t = np.exp((np.tile(u_t, nJ) + np.tile(v_t, nI).T - C) / eps)

            a_t = np.ones([nI, 1])  # Not really needed
            b_t = np.ones([nJ, 1])
            print('it = %d , eps = %f' % (it, eps))

    R = (np.tile(a_t, nJ) * K_t) * np.tile(b_t, nI).T  #  Reconstruct map

    return R, a_t, b_t, primals, duals, pdgaps


def calc_transport_pot_sinkhorn(source, target, costs, reg_param=1.e-1) -> (np.ndarray, float, np.ndarray, np.ndarray):
    """
    Implementation for solving ot using sinkhorn, including log-domain stabilization
    Also works on Unbalanced data

    source(np.ndarray): The source distribution, p
    target(np.ndarray): The target distribution, q
    costs(np.ndarray): The cost matrix
    reg_param(float): Regularization parameter, epsilon in the literature
    """
    K_t: np.ndarray = np.exp(costs / (-reg_param))
    Transport_cost, logs = ot.sinkhorn(source, target, costs, reg=reg_param, log=True)
    # Transport_cost, logs = ot.bregman.sinkhorn_stabilized(source.flatten(), target.flatten(), costs, reg=reg_param, log=True)
    u: np.ndarray = logs['u'].flatten()
    v: np.ndarray = logs['v'].flatten()
    Transport_plan: np.ndarray = np.diag(u) @ K_t @ np.diag(v)

    return Transport_plan, Transport_cost, u, v


def calc_transport_pot_emd(source, target, costs) -> (np.ndarray, float):
    """
    Implementation for solving ot using emd
    Also works on Unbalanced data

    source(np.ndarray): The source distribution, p
    target(np.ndarray): The target distribution, q
    costs(np.ndarray): The cost matrix
    """
    Transport_plan = ot.emd(source.flatten(), target.flatten(), costs)
    Transport_cost = np.sum(Transport_plan * costs)

    return Transport_plan, Transport_cost


def calc_transport_ott_sinkhorn(source: np.ndarray, target: np.ndarray, costs: np.ndarray,
                                reg_param: float = 1.e-2):
    """
    Not working yet

    source(np.ndarray): The source distribution, p
    target(np.ndarray): The target distribution, q
    costs(np.ndarray): The cost matrix
    reg_param(float): Regularization parameter, epsilon in the literature
    """
    source = source.flatten()
    target = target.flatten()
    geom = pointcloud.PointCloud(source, target, epsilon=reg_param)
    prob = linear_problem.LinearProblem(geom, a=source, b=target)

    solver = sinkhorn.Sinkhorn(threshold=1e-9, max_iterations=1000, lse_mode=True)

    out = solver(prob)

    print('hello world')

    return out.matrix


def unbalanced_sinkhorn(alpha: np.ndarray, beta: np.ndarray, costs: np.ndarray, eps=1.e-1,
                        max_iter=1000, return_plan=False) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This is the slow way, since it does not use the matrix-vector product formulation. The upside of this approach is
    That it is more stable.
    TODO: implement the faster one, using these iterations
    For more information about the math, see the paper: https://arxiv.org/pdf/2211.08775.pdf
    Unbalanced Sinkhorn algorithm for solving unbalanced OT problems. outputs vectors f_i and g_j,
    equal to the optimal transport potentials of the UOT(p, q) problem.
    :param alpha: source distribution and weights, p = sum(alpha_i * delta_x_i, i = 1...n)
    :param beta: target distribution and weights, q = sum(beta_i * delta_y_i, i = 1...m)
    :param costs: cost matrix, C_ij = c(x_i, y_j) in R^{n x m}
    :param eps: regularization parameter
    :param max_iter: maximum number of iterations
    :param return_plan: whether to return the transport plan
    dimensions:
    alpha_i in R^n
    beta_j in R^m
    x_i in R^{N x d}
    y_j in R^{M x d}
    :return: transport plan, f_i, g_j
    """
    if eps == 0:
        raise ValueError('eps must be positive')
    f = np.zeros(beta.shape).flatten()
    g = np.zeros(alpha.shape).flatten()
    iters = 0

    while iters < max_iter:
        for j in range(len(g)):
            g[j] = - eps * logsumexp(np.log(alpha) + (f - costs[:, j]) / eps)
            g[j] = approx_phi('KL', eps, -g[j])
        for i in range(len(f)):
            f[i] = - eps * logsumexp(np.log(beta) + (g - costs[i, :]) / eps)
            f[i] = approx_phi('KL', eps, -f[i])
        iters += 1

    if return_plan:
        plan = np.zeros([alpha.shape[0], beta.shape[0]], dtype=np.float64)
        for i in range(alpha.shape[0]):
            for j in range(beta.shape[0]):
                plan[i, j] = np.exp((f[i] + g[j] - costs[i, j]) / eps) * alpha[i] * beta[j]
        return f, g, plan

    return f, g, None


def approx_phi(divergence: str, eps: float, p: np.ndarray, ro: float = 0.5):
    # TODO: explain the variables
    if divergence == 'Balanced':
        return p

    if divergence == 'KL':
        temp = 1 + (eps / ro)
        return p / temp

    if divergence == 'TV':
        if p < ro:
            return -ro
        elif -ro < p < ro:
            return p
        else:
            return ro

    else:
        raise ValueError('Divergence not supported')


# noinspection PyArgumentList
def create_d_phi(source: np.ndarray, target: np.ndarray, ro: float) -> cp.Variable:
    """
    Takes two probability distributions and calculates the KL divergence between them.
    :param source: The source distribution
    :param target: The target distribution
    :param ro: TODO: explain
    :return: returns the variable d_phi as a cvxpy variable so that we could optimize on it
    """
    first_exp = cp.sum(cp.multiply(source / target, cp.log(source / target)))
    second_exp = cp.sum(source / target)
    D_phi = ro * cp.sum(first_exp - second_exp + 1)
    return D_phi


def split_signed_measure(source: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    This function splits the source measure into positive and negative parts.
    :param source: distribution to split
    :return: positive and negative part of the distribution
    """
    source_pos: np.ndarray = np.zeros(source.shape)
    source_neg: np.ndarray = np.zeros(source.shape)

    source_pos[source > 0] = source[source > 0]
    source_neg[source < 0] = -source[source < 0]

    return source_pos, source_neg


def is_degenerate(C: np.ndarray) -> bool:
    """
    Checks whether a matrix is degenerate.
    Args:
      C: A NumPy array.
    Returns:
      True if the matrix is degenerate, False otherwise.
    """
    # Check the sum of the costs in each row and column.
    for row in range(C.shape[0]):
        if np.sum(C[row, :]) == 0:
            return True
    for col in range(C.shape[1]):
        if np.sum(C[:, col]) == 0:
            return True

    # The matrix is not degenerate.
    return False


def is_valid_transport_plan(Plan: np.ndarray, p: np.ndarray, q: np.ndarray, tol=1e-6) -> bool:
    """
    Checks whether a matrix is a valid transport plan.
    Args:
      Plan: A NumPy array. The transport plan.
      p: A NumPy array. The source distribution, the sum of each row of Plan.
      q: A NumPy array. The target distribution, the sum of each column of Plan.
      tol: A float. The tolerance for checking the validity of the transport
    Returns:
      True if the matrix is a valid transport plan, False otherwise.
    """
    # Check that the rows and columns sum to the marginals.
    if not np.allclose(np.sum(Plan, axis=0), q, atol=tol):  # The sum of every column, adding up to q
        return False
    if not np.allclose(np.sum(Plan, axis=1), p, atol=tol):  # The sum of every row, adding up to p
        return False

    # The matrix is a valid transport plan.
    return True


def noise_image(im, noise_param=1e-2):
    """takes an image and adds noise to it"""
    noisy_image = np.copy(im)  # make a copy of the image, so that the original won't be changed
    height, width = im.shape
    for i in range(height):
        for j in range(width):
            noisy_image[i, j] += np.random.normal(0, noise_param)

    return noisy_image


def calculate_costs(size):
    """
    This function of an array or image and calculates the cost from it to itself.

    Parameters:
    - `size` (int or tuple): representing the object on which we would like to calculate costs.

    Returns:
    - `costs` (numpy.ndarray): A 2D array representing the matrix of costs of transporting pixels
                                from the first image to the second image.
    """
    # 1D case:
    if type(size) == int:
        # Generate an array of indices from 0 to size-1
        indices = np.arange(size)

        # Use broadcasting to calculate the absolute differences between each pair of indices
        costs = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])

        return costs

    # 2D case:
    elif len(size) == 2:
        # Extract the dimensions from the size tuple
        m, n = size
        size_1d = m * n

        # Generate coordinates for the grid
        coords = np.array([[i, j] for i in range(m) for j in range(n)])

        # Calculate the distance between each pair of points using broadcasting
        delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        delta_sq = delta ** 2

        # Compute the Euclidean distance by summing the squared differences and taking the square root
        distances = np.sqrt(np.sum(delta_sq, axis=2)).flatten()

        # Reshape the 1D distances into a 2D cost matrix
        costs = distances.reshape((size_1d, size_1d))

        return costs
