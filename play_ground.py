import numpy as np
import ot
import time
import matplotlib.pyplot as plt
import sys

sys.path.append('C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/Thesis/utils')
from utils.utils import *
from utils.Visualizations import *
from scipy.stats import norm, sem, t
from ot.datasets import make_1D_gauss as gauss

res = 100
lin_space = np.linspace(0, 1, res)
x = gauss(res, 35, 10)
y = gauss(res, 65, 10)

M = ot.dist(lin_space.reshape((res, 1)), lin_space.reshape((res, 1)),
            metric='euclidean')  # can be euclidean for L1 and sqeuclidean for L2
M /= M.max()

noise_values = np.logspace(start=-4, stop=1, num=31)
noise_param = noise_values[0]
x_noised, x_pos, x_neg = noise_and_split(x, noise_param)
y_noised, y_pos, y_neg = noise_and_split(y, noise_param)


def perform_noise_and_transport_analysis_wasserstein(p_normal, q_normal, p_unif, q_unif, noise, num_samples,
                                                     wasserstein_p=1):
    """
    Perform the noise and transport analysis for the Wasserstein distance.
    :param p_normal: source distribution
    :param q_normal: target distribution
    :param p_unif: source distribution
    :param q_unif: target distribution
    :param C: cost matrix
    :param noise: noise parameter
    :param num_samples: number of samples
    :param wasserstein_p: p parameter for the Wasserstein distance
    :return: dictionary with the results
    """
    results = {
        'classic_normal': [],
        'noised_normal': [],
        'ratios_emd_normal': [],
        'classic_unif': [],
        'noised_unif': [],
        'ratios_emd_unif': [],
        'linear': [],
        'linear_noised': [],
        'ratios_linear': [],
    }

    for i in range(num_samples):
        ppf_p_normal = np.cumsum(p_normal)
        ppf_q_normal = np.cumsum(q_normal)

        ppf_p_unif = np.cumsum(p_unif)
        ppf_q_unif = np.cumsum(q_unif)

        # Normal
        p_normal_noised, p_normal_pos, p_normal_neg = noise_and_split(p_normal, noise)
        q_normal_noised, q_normal_pos, q_normal_neg = noise_and_split(q_normal, noise)
        p_normal_post, q_normal_post = prep_signed_measures(p_normal_pos, p_normal_neg, q_normal_pos, q_normal_neg)
        ppf_p_normal_post = np.cumsum(p_normal_post)
        ppf_q_normal_post = np.cumsum(q_normal_post)

        # Uniform
        p_unif_noised, p_unif_pos, p_unif_neg = noise_and_split(p_unif, noise)
        q_unif_noised, q_unif_pos, q_unif_neg = noise_and_split(q_unif, noise)
        p_unif_post, q_unif_post = prep_signed_measures(p_unif_pos, p_unif_neg, q_unif_pos, q_unif_neg)
        ppf_p_unif_post = np.cumsum(p_unif_post)
        ppf_q_unif_post = np.cumsum(q_unif_post)

        # Compute transport
        W_distance_classic_normal = ot.wasserstein_1d(ppf_p_normal, ppf_q_normal, p=wasserstein_p)
        W_distance_noised_normal = ot.wasserstein_1d(ppf_p_normal_post, ppf_q_normal_post, p=wasserstein_p)

        W_distance_classic_unif = ot.wasserstein_1d(ppf_p_unif, ppf_q_unif, p=wasserstein_p)
        W_distance_noised_unif = ot.wasserstein_1d(ppf_p_unif_post, ppf_q_unif_post, p=wasserstein_p)

        results['classic_normal'].append(W_distance_classic_normal)
        results['noised_normal'].append(W_distance_noised_normal)
        results['ratios_emd_normal'].append(W_distance_noised_normal / W_distance_classic_normal)

        results['classic_unif'].append(W_distance_classic_unif)
        results['noised_unif'].append(W_distance_noised_unif)
        results['ratios_emd_unif'].append(W_distance_noised_unif / W_distance_classic_unif)

        # Linear
        results['linear'].append(np.linalg.norm(p_normal - q_normal))
        results['linear_noised'].append(np.linalg.norm(p_normal_noised - q_normal_noised))
        results['ratios_linear'].append(np.linalg.norm(p_normal_noised - q_normal_noised) / np.linalg.norm(p_normal - q_normal))

    # Compute mean and std
    results['mean_normal'], results['ci_normal'] = confidence_interval(results['classic_normal'])
    results['mean_normal_noised'], results['ci_normal_noised'] = confidence_interval(results['noised_normal'])

    results['mean_unif'], results['ci_unif'] = confidence_interval(results['classic_unif'])
    results['mean_unif_noised'], results['ci_unif_noised'] = confidence_interval(results['noised_unif'])

    return results
