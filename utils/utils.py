import numpy as np
from Classes import TransportResults
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import ot
import jax.numpy as jnp
from ot.datasets import make_1D_gauss as gauss
from scipy.special import logsumexp
import pandas as pd
import cvxpy as cp
from scipy.stats import norm, sem, t


def solve_ot_dual(c, mu, nu):
    n, m = c.shape  # n and m are the dimensions of cost matrix c
    phi = cp.Variable(n)
    psi = cp.Variable(m)

    constraints = [phi[i] + psi[j] <= c[i, j] for i in range(n) for j in range(m)]

    objective = cp.Maximize(mu @ phi + nu @ psi)
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return phi.value, psi.value, problem.value


def create_constraints_general(case, source, target):
    """
    Creates a matrix variable and a set of constraints based on the specified case.

    Parameters:
    - case (str): The case to use ('normal', 'lifted', or 'signed').
    - source (list): A list of non-negative numbers representing the source distribution.
    - target (list): A list of non-negative numbers representing the target distribution.

    Returns:
    - T_matrix (cp.Variable or tuple of cp.Variable): A matrix variable or tuple of variables representing the transport plan.
    - cons (list): A list of cvxpy constraints.
    - alpha (float, optional): Lifting parameter for 'lifted' case.
    """
    if case == 'normal':
        T_matrix = cp.Variable((len(source), len(target)), nonneg=True)
        cons = [cp.sum(T_matrix, axis=0) == target,
                cp.sum(T_matrix, axis=1) == source,
                T_matrix >= 0]

    elif case == 'lifted':
        T_matrix = cp.Variable((len(source), len(target)), nonneg=True)
        alpha = -min(min(source), min(target)) + 1
        cons = [cp.sum(T_matrix, axis=0) == target + alpha,
                cp.sum(T_matrix, axis=1) == source + alpha,
                T_matrix >= 0]
        return T_matrix, alpha, cons

    elif case == 'signed':
        T_matrix_pos = cp.Variable((len(source), len(target)), nonneg=True)
        T_matrix_neg = cp.Variable((len(source), len(target)), nonneg=True)
        cons = [cp.sum(T_matrix_pos - T_matrix_neg, axis=0) == target,
                cp.sum(T_matrix_pos - T_matrix_neg, axis=1) == source,
                T_matrix_pos >= 0,
                T_matrix_neg >= 0]
        return T_matrix_pos, T_matrix_neg, cons

    else:
        raise ValueError("Invalid case specified")

    return T_matrix, cons


def create_T(source: np.ndarray, target: np.ndarray, cost_matrix: np.ndarray, transport_type: str = 'standard'):
    """
    This function takes a TransportResults object and updates it according to the transport type.

    """
    if transport_type == 'standard':
        p, constraints = create_constraints_general('normal', source, target)
        obj = cp.Minimize(cp.sum(cp.multiply(p, cost_matrix)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return TransportResults(transported_mass=prob.value, transport_plan=p.value,
                                source_distribution=source, target_distribution=target)

    elif transport_type == 'lifted':
        T, alpha, constraints = create_constraints_general('lifted', source.flatten(), target.flatten())
        obj = cp.Minimize(cp.sum(cp.multiply(T, cost_matrix)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        return TransportResults(transported_mass=prob.value, transport_plan=T.value, lift_parameter=alpha,
                                source_distribution=source, target_distribution=target)

    elif transport_type == 'signed':
        T_pos, T_neg, constraints = create_constraints_general('signed', source.flatten(), target.flatten())
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


def split_signed_measure(src: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    This function splits the source measure into positive and negative parts.
    :param src: distribution to split
    :return: positive and negative part of the distribution
    """
    source = np.copy(src)
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


def calculate_costs(size, distance_metric='L1'):
    """
    This function of an array or image and calculates the cost from it to itself.

    Parameters:
    - `size` (int or tuple): representing the object on which we would like to calculate costs.

    Returns:
    - `costs` (numpy.ndarray): A 2D array representing the matrix of costs of transporting pixels
                                from the first image to the second image.
    """

    # Helper function for L1 and L2 distance
    if distance_metric == 'L1':
        dist = lambda a, b: abs(a - b)
    elif distance_metric == 'L2':
        dist = lambda a, b: (a - b) ** 2
    else:
        raise ValueError('Invalid distance metric. Must be either "L1" or "L2".')

    # 1D case:
    if isinstance(size, int):
        X = np.linspace(0, 1, size)
        costs = np.zeros([size, size], np.float64)

        for it1 in range(size):
            for it2 in range(size):
                costs[it1, it2] = dist(X[it1], X[it2])

        return costs

    # 2D case:
    elif len(size) == 2:
        m, n = size
        size_1d = m * n

        coords = np.array([[i, j] for i in range(m) for j in range(n)])
        delta = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

        if distance_metric == 'L1':
            distances = np.sum(np.abs(delta), axis=2)
        elif distance_metric == 'L2':
            distances = np.sqrt(np.sum(delta ** 2, axis=2))
        else:
            raise ValueError('Invalid distance metric. Must be either "L1" or "L2".')

        costs = distances.reshape((size_1d, size_1d))

        return costs


def run_experiment_and_append(df, p, q, res, SNR, scale_param, num_samples=100, distance_metric='L2') -> pd.DataFrame:
    """
    This function runs an experiment and appends the results to a DataFrame.
    df (pandas.DataFrame): The DataFrame to append the results to.
    p (numpy.ndarray): The source distribution.
    q (numpy.ndarray): The target distribution.
    res (int): The resolution of the distributions.
    SNR (float): The signal-to-noise ratio.
    scale_param (float): The scale parameter.
    num_samples (int): The number of samples to run the experiment on.
    distance_metric (str): The distance metric to use.
    """

    signal_power = (p ** 2).sum()
    noise = noise_from_SNR(SNR, signal_power=signal_power, res=res)

    C = get_distance_matrix(res, distance_metric)

    results = perform_noise_and_transport_analysis(p, q, C, noise, num_samples=num_samples)

    # Create new row
    new_row = {
        'Res': res,
        'Noise_Param': noise,
        'Scale_Param': scale_param,
        'Signal_Power': signal_power,
        'SNR': SNR,
        'Distance_Classic': results['mean_classic'],
        'CI_Distances_Classic': results['ci_classic'],
        'Distances_Noised': results['mean_noised'],
        'CI_Distances_Noised': results['ci_noised'],
        'Ratios_EMD': np.mean(results['ratios_emd']),
        'Distances_Linear':  np.mean(results['results_linear']),
        'Distances_Linear_Noised': np.mean(results['results_linear_noised']),
        'Ratios_Linear': np.mean(results['ratios_linear'])
    }

    # Append new row to DataFrame
    return df._append(new_row, ignore_index=True)


def perform_noise_and_transport_analysis(p, q, C, noise, num_samples=100) -> dict:
    results = {
        'classic': [],
        'noised': [],
        'ratios_emd': [],
        'linear': [],
        'linear_noised': [],
        'ratios_linear': []
    }

    for i in range(num_samples):
        p_noised, p_pos, p_neg = noise_and_split(p, noise)
        q_noised, q_pos, q_neg = noise_and_split(q, noise)

        p_post, q_post = prep_signed_measures(p_pos, p_neg, q_pos, q_neg)

        _, results_classic_add = calc_transport_pot_emd(p, q, C)
        _, results_noised_add = calc_transport_pot_emd(p_post, q_post, C)

        results['classic'].append(results_classic_add)
        results['noised'].append(results_noised_add)
        results['ratios_emd'].append(results_classic_add / results_noised_add)

        results['linear'].append(np.linalg.norm(p - q))
        results['linear_noised'].append(np.linalg.norm(p_noised - q_noised))
        results['ratios_linear'].append(np.linalg.norm(p - q) / np.linalg.norm(p_noised - q_noised))

    # Calculate confidence intervals for the results
    results['mean_classic'], results['ci_classic'] = confidence_interval(results['classic'])
    results['mean_noised'], results['ci_noised'] = confidence_interval(results['noised'])

    return results


def run_experiment_and_append_images(df, im1, im2, SNR, n_samples=100):
    results_classic = []
    results_noised = []
    ratios_emd = []
    results_linear = []
    results_linear_noised = []
    ratios_linear = []

    noise_param = noise_from_SNR(SNR, signal_power=im1.sum(), res=im1.shape[0])

    for i in range(n_samples):
        im1_noised, im2_noised, im1_post, im2_post, C = create_images_and_costs(im1_base=im1, im2_base=im2,
                                                                                noise=noise_param)

        results_classic_add = calc_transport_pot_emd(im1.flatten(), im2.flatten(), C)[1]
        results_noised_add = calc_transport_pot_emd(im1_post.flatten(), im2_post.flatten(), C)[1]

        results_classic.append(results_classic_add)
        results_noised.append(results_noised_add)
        ratios_emd.append(results_classic_add / results_noised_add)

        results_linear.append(np.linalg.norm(im1 - im2))
        results_linear_noised.append(np.linalg.norm(im1_noised - im2_noised))
        ratios_linear.append(np.linalg.norm(im1 - im2) / np.linalg.norm(im1_noised - im2_noised))

    mean_noised, ci_noised = confidence_interval(results_noised)
    mean_linear_noised, ci_linear_noised = confidence_interval(results_linear_noised)

    new_row = {
        'SNR': SNR,
        'Noise_Param': noise_param,
        'Im_Size': im1.shape[0],  # Assuming square images
        'Distances_Classic': np.mean(results_classic),
        'Distances_Noised': mean_noised,
        'CI_Distances_Noised': ci_noised,
        'Ratios_EMD': np.mean(ratios_emd),
        'Distances_Linear': np.mean(results_linear),
        'Distances_Linear_Noised': mean_linear_noised,
        'CI_Distances_Linear_Noised': ci_linear_noised,
        'Ratios_Linear': np.mean(ratios_linear)
    }

    # Append new row to DataFrame
    return df._append(new_row, ignore_index=True)


def create_distribs_and_costs(res, noise, scale_parameter=1, distance_metric='L2', first_center=0.35, first_std=0.1,
                              second_center=0.65, second_std=0.1):
    X = np.linspace(0, scale_parameter, res)
    p = norm.pdf(X, scale_parameter * first_center, scale_parameter * first_std)
    p = p / p.sum()
    q = norm.pdf(X, scale_parameter * second_center, scale_parameter * second_std)
    q = q / q.sum()

    C = get_distance_matrix(res=res, distance_metric=distance_metric)

    p_noised, p_pos, p_neg = noise_and_split(p, noise)
    q_noised, q_pos, q_neg = noise_and_split(q, noise)

    p_post, q_post = prep_signed_measures(p_pos, p_neg, q_pos, q_neg)

    return p, q, p_noised, q_noised, p_post, q_post, C


def create_images_and_costs(im1_base, im2_base, noise, distance_metric='L1'):
    """
    This function creates two 1D distributions and a cost matrix between them.
    :param im2_base: The initial image 2
    :param im1_base: The initial image 1
    :param noise: noise parameter
    :param distance_metric: distance metric to use
    :return: p, q, C
    """
    C = calculate_costs(im1_base.shape, distance_metric=distance_metric)

    im1_noised = noise_image(im1_base, noise)
    im2_noised = noise_image(im2_base, noise)

    im1_pos, im1_neg = split_signed_measure(im1_noised)
    im2_pos, im2_neg = split_signed_measure(im2_noised)

    im1_post = im1_pos + im2_neg
    im2_post = im1_neg + im2_pos

    mean_distribs = (im2_post.sum() + im1_post.sum()) / 2
    im1_post = im1_post * (mean_distribs / im1_post.sum())
    im2_post = im2_post * (mean_distribs / im2_post.sum())

    return im1_noised, im2_noised, im1_post, im2_post, C


def confidence_interval(data, confidence=0.96):
    # Compute the sample mean
    mean = np.mean(data)

    # Compute the standard error of the mean
    std_err = sem(data)

    # Get the degrees of freedom and lookup the t-value
    dof = len(data) - 1
    t_val = t.ppf((1 + confidence) / 2, dof)

    # Compute the margin of error
    margin_error = t_val * std_err

    return mean, margin_error


def noise_from_SNR(SNR, signal_power, res):
    """
    This function calculates the noise power from the SNR and the signal power. and decides what the noise parameter
    is according to the power. The noise is assumed to be Gaussian.
    :param res:
    :param SNR:
    :param signal_power:
    :return:
    """
    noise_power = signal_power / SNR
    noise_param = np.sqrt(noise_power / res)
    return noise_param


def noise_and_split(dist, noise_param):
    """
    This function takes a distribution and a given noise parameter and returns
    The noised distribution, its positive part, and its negative part.
    :param dist:
    :param noise_param:
    :return:
    """
    noise_p = np.random.normal(0, noise_param, len(dist))

    p_noised = dist + noise_p

    p_pos, p_neg = split_signed_measure(p_noised)

    return p_noised, p_pos, p_neg


def prep_signed_measures(p_pos, p_neg, q_pos, q_neg):
    p_post = p_pos + q_neg
    q_post = p_neg + q_pos

    mean_distribs = (q_post.sum() + p_post.sum()) / 2
    p_post = p_post * (mean_distribs / p_post.sum())
    q_post = q_post * (mean_distribs / q_post.sum())

    return p_post, q_post


def get_distance_matrix(res, distance_metric='L2'):

    X = np.linspace(0, 1, res)
    C = np.zeros([res, res], dtype=np.float64)
    if distance_metric == 'L1':
        dist = lambda a, b: abs(a - b)
    elif distance_metric == 'L2':
        dist = lambda a, b: (a - b) ** 2
    else:
        raise ValueError('Invalid distance metric. Must be either "L1" or "L2".')

    for it1 in range(res):
        for it2 in range(res):
            C[it1, it2] = dist(X[it1], X[it2])

    return C


def perform_noise_and_transport_analysis_wasserstein(p, q, C, noise, num_samples, wasserstein_p=1):
    """
    Perform noise and transport analysis for the Wasserstein distance
    :param wasserstein_p:
    :param p: source distribution
    :param q: target distribution
    :param C: cost matrix
    :param noise: noise parameter
    :param num_samples: number of samples
    :return: noise, transport, noise_std, transport_std
    """
    results = {
        'classic': [],
        'noised': [],
        'ratios_emd': [],
        'linear': [],
        'linear_noised': [],
        'ratios_linear': []
    }

    for i in range(num_samples):
        p /= p.sum()
        q /= q.sum()
        ppf_p = np.cumsum(p)
        ppf_q = np.cumsum(q)

        p_noised, p_pos, p_neg = noise_and_split(p, noise)
        q_noised, q_pos, q_neg = noise_and_split(q, noise)

        p_post, q_post = prep_signed_measures(p_pos, p_neg, q_pos, q_neg)

        # p_post /= p_post.sum()
        # q_post /= q_post.sum()
        ppf_p_post = np.cumsum(p_post)
        ppf_q_post = np.cumsum(q_post)

        W_distance_classic = ot.wasserstein_1d(ppf_p, ppf_q, p=wasserstein_p)
        W_distance_noised = ot.wasserstein_1d(ppf_p_post, ppf_q_post, p=wasserstein_p)

        results['classic'].append(W_distance_classic)
        results['noised'].append(W_distance_noised)
        results['ratios_emd'].append(W_distance_noised / W_distance_classic)

        # Linear
        results['linear'].append(np.linalg.norm(p - q))
        results['linear_noised'].append(np.linalg.norm(p_noised - q_noised))
        results['ratios_linear'].append(np.linalg.norm(p_noised - q_noised) / np.linalg.norm(p - q))

    # Compute mean and std
    results['mean_classic'], results['ci_classic'] = confidence_interval(results['classic'])
    results['mean_noised'], results['ci_noised'] = confidence_interval(results['noised'])

    return results