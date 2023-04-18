import os
import ot
import numpy as np
import pathlib
import cv2
from utils.Kantorivich_problem import solve_kantorovich, calculate_costs


def normalize_and_flatten(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float64)
    im_norm = im / sum(sum(im))
    im_norm = im_norm.flatten()
    return im_norm


current_path = os.getcwd()
current_path_parent = os.path.dirname(current_path)

im1_path = current_path_parent + '\\images\\im1_test.png'
im2_path = current_path_parent + '\\images\\im2_test.png'

# im1_path = current_path_parent + '\\images\\mnist_image_1.jpg'
# im2_path = current_path_parent + '\\images\\image_left.jpg'

im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)

im1_norm = normalize_and_flatten(im1)
im2_norm = normalize_and_flatten(im2)

costs = calculate_costs(len(im1.flatten()))

dist, logs = ot.emd2(im1_norm, im2_norm, costs, log=True, return_matrix=True)
transport_mat = logs['G']
write_path = r'D:\Erik\Optimal_Transport\Thesis\images\test.png'
transport_mat = 255 * transport_mat / transport_mat.max()
cv2.imwrite(write_path, transport_mat)
print('hello world')
