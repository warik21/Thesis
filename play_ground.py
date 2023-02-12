import timeit
import ot
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from utils.Kantorivich_problem import *


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

# source = cv2.imread(r'C:\Users\eriki\Documents\school\Thesis\Optimal_transport_playground\IMG_2005.jpg')
# target = cv2.imread(r'C:\Users\eriki\Documents\school\Thesis\Optimal_transport_playground\20220819_181050.jpg')
source = cv2.imread(r'C:\Users\eriki\Documents\school\Thesis\Optimal_transport_playground\mnist_image_1.jpg', cv2.IMREAD_GRAYSCALE)
# target = cv2.imread(r'C:\Users\eriki\Documents\school\Thesis\Optimal_transport_playground\image_left.jpg')
target = image_2_pixels_left(source)
# source = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)

# source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
# target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

costs, source_im_1d, target_im_1d = calculate_costs(source, target)

min_cost, transport_matrix = solve_transport(source_im_1d, target_im_1d, costs)

# Create histograms for the images
hist1, _ = np.histogram(source, bins=256, range=(0, 1))
hist2, _ = np.histogram(target, bins=256, range=(0, 1))

# Reshape the histograms to 2D arrays
hist1 = hist1.reshape(-1, 1)
hist2 = hist2.reshape(-1, 1)

# Normalize the histograms
hist1 = hist1 / np.sum(hist1)
hist2 = hist2 / np.sum(hist2)

# Define the cost matrix
cost_matrix = ot.dist(hist1, hist2, metric='euclidean')

# Compute the regularized OT distance
ot_distance = ot.emd2(hist1, hist2, cost_matrix)

# Print the OT distance
print("OT distance: {}".format(ot_distance))
##Calculate the distance between the images:


# solve_ott = jax.jit(solve_ott, backend='gpu')
# out = solve_ott(X1, X2)
# print('blabla')