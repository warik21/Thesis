import sys

import jax.numpy as jnp
from jax import random
from ot.datasets import make_1D_gauss as gauss
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear.univariate import UnivariateSolver

sys.path.append('C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/Thesis/utils')
from utils.utils import *


class AbsoluteDifferenceCost(costs.TICost):
    def h(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(x)


class SquareDifferenceTICost(costs.TICost):
    def h(self, x: jnp.ndarray) -> jnp.ndarray:
        return x ** 2


def run_experiment_and_append(df, res, SNR, num_samples=100, wasserstein_p=1,
                              first_center=0.35, first_std=0.1,
                              second_center=0.65, second_std=0.1) -> pd.DataFrame:
    # Generate Gaussian distributions
    p = gauss(res, m=res * first_center, s=res * first_std)
    q = gauss(res, m=res * second_center, s=res * second_std)

    # Calculate signal power and determine noise level
    signal_power = (p ** 2).sum()
    noise_level = noise_from_SNR(SNR, signal_power=signal_power, res=res)

    # Compute original and noised Wasserstein distances
    original_distance = wasserstein_distance(p, q, wasserstein_p)

    # Add noise to the distributions
    key1, key2 = random.split(random.PRNGKey(0), 2)
    noisy_p = p + noise_level * random.normal(key1, p.shape)
    noisy_q = q + noise_level * random.normal(key2, q.shape)
    noised_distance = wasserstein_distance(noisy_p, noisy_q, wasserstein_p)

    p_pos = noisy_p[noisy_p >= 0]
    p_neg = noisy_p[noisy_p < 0]
    q_pos = noisy_q[noisy_q >= 0]
    q_neg = noisy_q[noisy_q < 0]

    p_post = p_pos + q_neg
    q_post = q_pos + p_neg

    post_process_distance = wasserstein_distance(p_post, q_post, wasserstein_p)

    # Prepare the data to be appended
    new_row = {
        'Res': res,
        'SNR': SNR,
        'Signal_Power': signal_power,
        'Noise_Level': noise_level,
        'Wasserstein_p': wasserstein_p,
        'Distance_Original': original_distance,
        'Distance_Noised': noised_distance,
        'Distance_Post': post_process_distance,
        'Ratio': original_distance / noised_distance
    }

    # Append new data to DataFrame
    return df._append(new_row, ignore_index=True)


def compute_cdf(arr):
    # Normalize the array to sum to 1 and compute the cumulative sum to get the CDF
    return jnp.cumsum(arr) / jnp.sum(arr)


class PWassersteinCost(costs.TICost):
    def __init__(self, p_value=1):
        self.p_value = p_value

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Compute the p-th power of absolute differences
        return jnp.abs(x - y) ** self.p_value


def wasserstein_distance(p, q, wasserstein_p=1):
    # Compute the CDFs for p and q
    cdf_p = compute_cdf(p)
    cdf_q = compute_cdf(q)

    # Function to compute the Wasserstein distance
    if wasserstein_p == 1:
        cost_fn = AbsoluteDifferenceCost()  # Use the appropriate cost function here
    elif wasserstein_p == 2:
        cost_fn = SquareDifferenceTICost()
    else:
        raise ValueError('Invalid distance metric, only supporting W1 and W2 at the moment')
    geom = pointcloud.PointCloud(cdf_p[:, jnp.newaxis], cdf_q[:, jnp.newaxis], cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)
    solver = UnivariateSolver()
    result = solver(prob)

    # Return the Wasserstein distance raised to the power of 1/wasserstein_p
    if wasserstein_p == 1:
        return result.ot_costs.sum()
    elif wasserstein_p == 2:
        return jnp.sqrt(result.ot_costs.sum())
