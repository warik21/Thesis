import numpy as np
import sys

sys.path.append('C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/Thesis/utils')
import ot
from scipy.special import logsumexp
import pandas as pd
# import cvxpy as cp
from scipy.stats import norm, sem, t
import time
from typing import Tuple
from pydantic import BaseModel
from typing import Optional


def calc_transport_pot_emd(source, target, costs) -> (np.ndarray, float):  # type: ignore
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


class TransportResults(BaseModel):
    transport_plan: np.ndarray  # shape=(dim_source, dim_target)
    source_distribution: np.ndarray  # shape=(dim_source,)
    target_distribution: np.ndarray  # shape=(dim_target,)
    transported_mass: float  # transported mass, sum of transport_plan, the same as cost.
    lift_parameter: Optional[float] = None  # lift, the amount by which we lifted the distributions
    Pos_plan: Optional[np.ndarray] = None  # shape=(dim_source, dim_target)
    Neg_plan: Optional[np.ndarray] = None  # shape=(dim_source, dim_target)

    class Config:
        arbitrary_types_allowed = True


def split_signed_measure(src: np.ndarray) -> (np.ndarray, np.ndarray):  # type: ignore
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
        I, J = np.indices(size)

        # Flatten the indices to create 1D arrays of x and y coordinates
        I_flat = I.flatten()
        J_flat = J.flatten()

        # Calculate distances using broadcasting
        if distance_metric == 'L1':
            costs = np.sqrt((I_flat[:, None] - I_flat[None, :]) ** 2 + (J_flat[:, None] - J_flat[None, :]) ** 2)
        elif distance_metric == 'L2':
            costs = (I_flat[:, None] - I_flat[None, :]) ** 2 + (J_flat[:, None] - J_flat[None, :]) ** 2
        else:
            raise ValueError('Invalid distance metric. Must be either "L1" or "L2".')

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
        'Distances_Linear': np.mean(results['linear']),
        'Distances_Linear_Noised': np.mean(results['linear_noised']),
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


def run_experiment_and_append_images(df, im1, im2, SNR, distance_metric='L2', n_samples=100):
    """
    This function runs an experiment and appends the results to a DataFrame.
    :param distance_metric:
    :param df:
    :param im1:
    :param im2:
    :param SNR:
    :param n_samples:
    :return:
    """
    results_classic = []
    results_noised = []
    ratios_emd = []
    results_linear = []
    results_linear_noised = []
    ratios_linear = []

    noise_param = noise_from_SNR(SNR, signal_power=im1.sum(), res=im1.shape[0])

    for i in range(n_samples):
        im1_noised, im2_noised, im1_post, im2_post, C = create_images_and_costs(im1_base=im1, im2_base=im2,
                                                                                noise=noise_param,
                                                                                distance_metric=distance_metric)

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
    :param im2_base(np.ndarray): The second image to compare.
    :param im1_base(np.ndarray): The first image to compare.
    :param noise(float): The noise parameter.
    :param distance_metric(str): The distance metric to use.
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
    
    Args:
        SNR (float): The signal-to-noise ratio.
        signal_power (float): The power of the signal.
        res (int): The resolution of the signal.

    Returns:
        noise_param (float): The noise parameter.
    """
    noise_power = signal_power / SNR
    noise_param = np.sqrt(noise_power / (res ** 2))
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


def perform_noise_and_transport_analysis_wasserstein(p, q, noise, num_samples, wasserstein_p=1):
    """
    Perform noise and transport analysis for the Wasserstein distance
    :param wasserstein_p:
    :param p: source distribution
    :param q: target distribution
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


def fourier_transform_shift(im) -> np.ndarray:
    """
    This function takes an image and returns its Fourier transform.
    :param im(np.ndarray): The image to transform (2D array).
    :return: dft_mag_shifted(np.ndarray): The shifted magnitude of the Fourier transform.
    """
    im = np.copy(im)
    # Normalize
    im = im / im.sum()
    # Fourier transform
    DFT_im = np.fft.fft2(im)
    DFT_mag = np.abs(DFT_im)
    DFT_mag_shifted = np.fft.fftshift(DFT_mag)

    return DFT_mag_shifted


def Fourier1(mu, nu, T=2 * np.pi) -> float:
    mu_hat = np.fft.fft2(mu)
    nu_hat = np.fft.fft2(nu)
    m, n = mu.shape
    dxdy = (m / T) * (n / T)

    integral = 0

    for y in range(n):
        for x in range(m):
            if x == 0 and y == 0:
                continue

            kx = x * (2 * np.pi / m)
            ky = y * (2 * np.pi / n)

            # Compute the squared magnitude of the frequency vector
            k_squared = kx ** 2 + ky ** 2

            # Increment the integral value
            diff = mu_hat[y, x] - nu_hat[y, x]
            integral += (np.abs(diff) ** 2) / k_squared * dxdy

    integral = np.sqrt(((1 / T) ** 2) * integral)

    return integral


def Fourier2(a, b, T=2 * np.pi) -> float:
    m, n = np.shape(a)
    dxdy = (T / m) * (T / n)

    # Normalize a and b
    a /= np.sum(a)
    b /= np.sum(b)

    # Calculate expected values and translation vector efficiently
    expected_value_a = np.array([np.sum(a * np.arange(m)[:, None]), np.sum(a * np.arange(n))])
    expected_value_b = np.array([np.sum(b * np.arange(m)[:, None]), np.sum(b * np.arange(n))])
    translation_vector = expected_value_a - expected_value_b

    # Perform FFT
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    # Calculate distance considering the translation vector
    integral = 0
    for y in range(n):
        for x in range(m):
            # Avoid division by zero for the zero frequency component
            if x == 0 and y == 0:
                continue

            kx = x * T / m
            ky = y * T / n

            k_squared = kx ** 2 + ky ** 2
            trasl = np.exp((2 * np.pi * 1j * translation_vector[0] * x) / m) * np.exp(
                (2 * np.pi * 1j * translation_vector[1] * y) / n)
            integral += ((np.abs(fa[x, y] - fb[x, y] * trasl)) ** 2) / (k_squared ** 2) * dxdy

    C = (((1 / T) ** 2) * integral) ** (1 / 2)  # This is essentially the difference between a and b_moved
    distance = np.sqrt((C ** 2) + (np.sum(translation_vector ** 2)))  # Here we add the translation vector and normalize

    return distance


def calculate_and_time_wasserstein(image1, image2, cost_matrix) -> Tuple[float, float]:
    start_time = time.time()
    distance = ot.emd2(image1.flatten(), image2.flatten(), cost_matrix)
    elapsed_time = time.time() - start_time
    return distance, elapsed_time


def calculate_and_time_fourier1(image1, image2) -> Tuple[float, float]:
    start_time = time.time()
    distance = Fourier1(image1, image2)
    elapsed_time = time.time() - start_time
    return distance, elapsed_time


def calculate_and_time_fourier2(image1, image2) -> Tuple[float, float]:
    start_time = time.time()
    distance = Fourier2(image1, image2)
    elapsed_time = time.time() - start_time
    return distance, elapsed_time


def calculate_and_time_l2(image1, image2) -> Tuple[float, float]:
    start_time = time.time()
    distance = np.linalg.norm(image1 - image2)
    elapsed_time = time.time() - start_time
    return distance, elapsed_time


def calculate_and_time_UOT(image1, image2, cost_matrix, reg=1e-3, reg_m=1e-3) -> Tuple[float, float]:
    start_time = time.time()
    distance = ot.unbalanced.sinkhorn_unbalanced(image1.flatten(), image2.flatten(),
                                                 cost_matrix, reg, reg_m)
    elapsed_time = time.time() - start_time
    return distance, elapsed_time
