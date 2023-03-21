import numpy as np
import cvxpy as cp
import cv2
import time
import ot
import ott
from scipy.spatial.distance import cdist
"""
The script starts by importing the time library, which is used to measure the execution time of the script.

The script loads an image from a specific file path and reads it in grayscale mode using cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE).

The script makes a copy of the original image and adds a 2 pixel wide border to the right side of the image using cv2.copyMakeBorder(image_1, 0, 0, 2, 0, cv2.BORDER_CONSTANT, value=255).

The script then removes the 2 pixel wide border from the right side of the image using slicing img_added[:, :-2].

The script then writes the modified image to a new file called 'image_left.jpg' using cv2.imwrite('image_left.jpg', img_moved).

The script reshape the original image and the modified image into 1-D arrays using S = image_1.reshape(-1) and T = img_moved.reshape(-1)

The script calculates the transport cost between the two images using the transport_cost function, which takes three arguments: the reshaped original image (S), the reshaped modified image (T), and the cost matrix (costs). It returns the minimum cost and the transport matrix.

The script prints the minimum cost divided by the square root of the number of pixels in the image, and the transport matrix.

Finally, the script calculates the elapsed time of the script and prints it to the console.
"""


def create_constraints(source, target):
    """
    This function takes two lists as input and creates a matrix variable and a set of constraints.

    Parameters:
    - source (list): A list of non-negative numbers representing the source distribution.
    - target (list): A list of non-negative numbers representing the target distribution.

    Returns:
    - p_matrix (cvxpy.Variable): A matrix variable with shape (len(source), len(target)) representing the transport plan.
    - cons (list): A list of cvxpy constraints.

    Constraints:
    - The sum of each column of p_matrix is equal to the corresponding element of target.
    - The sum of each row of p_matrix is equal to the corresponding element of source.
    - p_matrix is element-wise non-negative.
    """
    p_matrix = cp.Variable((len(source), len(target)), nonneg=True)

    # noinspection PyTypeChecker
    cons = [cp.sum(p_matrix, axis=0) == target,  # column sum should be what we move from the pixel the column represents
            cp.sum(p_matrix, axis=1) == source,  # row sum should be what we move from the pixel the row represents
            p_matrix >= 0]  # all elements of p should be non-negative

    return p_matrix, cons

def solve_transport(source, target, cost_matrix):
    """
    This function takes two lists and a matrix as input and solves a linear transport problem.

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
    p, constraints = create_constraints(source, target)

    obj = cp.Minimize(cp.sum(cp.multiply(p, cost_matrix)))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return prob.value, p.value

def calculate_costs(size: int):
    """
    This function takes the size of the images, and calculates the costs of transporting
    pixels from one position to the other. Returns a mapping of euclidean distances.

    Parameters:
    - `size` (int): A 2D array representing the first image.

    Returns:
    - `costs` (numpy.ndarray): A 2D array representing the matrix of costs of transporting pixels
                                from the first image to the second image.
    """

    costs = np.zeros([size, size])

    m,n = int(np.sqrt(size)), int(np.sqrt(size))
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    location_x = (i * n) + j
                    location_y = (k * n) + l
                    costs[location_x, location_y] += np.sqrt((i - k) ** 2 + (j - l) ** 2)
    return costs

def image_2_pixels_left(im: np.ndarray):
    """
    This function takes an image, `im` and moves it 2 pixels to the left.

    Parameters:
    - `im` (numpy.ndarray): A 2D array representing the image.

    Returns:
    - `img_moved_left` (numpy.ndarray): A 2D array representing the image after moving it 2 pixels to the left.

    This function starts by creating a copy of the image with a 2 pixels wide white border on the left side using `cv2.copyMakeBorder(im, 0, 0, 2, 0, cv2.BORDER_CONSTANT, value=255)`.
    Then it removes the last 2 pixels from the left side of the image using `img_moved_left = img_moved_left[:, :-2]`
    Finally, it returns the image after moving it 2 pixels to the left.
    """
    img_moved_left = cv2.copyMakeBorder(im, 0, 0, 1, 0, cv2.BORDER_CONSTANT, value=255)
    img_moved_left = img_moved_left[:, :-1]
    return img_moved_left

def create_image(size: tuple, pixel_location: tuple):
    """
    Creates a Grayscale image with the specified size(all pixels black) and a white pixel at the specified location.

    Args:
        size: A tuple of the form (width, height) representing the size of the image in pixels.
        pixel_location: A tuple of the form (x, y) representing the location of the white pixel in the image.

    Returns:
        An image represented as a numpy array with shape (height, width) and dtype np.uint8.
        The white pixel will be at the specified location, and all other pixels will be black.
    """
    # Create a numpy array of zeros with the specified size
    img = np.zeros(size, dtype=np.uint8)

    # Set the specified pixel to white (255)
    img[pixel_location[1], pixel_location[0]] = 255

    # Return the image
    return img

def solve_kantorovich(im1: np.ndarray, im2: np.ndarray, costs:np.ndarray):
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    im1_norm = im1 / sum(sum(im1))
    im2_norm = im2 / sum(sum(im2))

    im1_norm = im1_norm.flatten()
    im2_norm = im2_norm.flatten()

    min_cost, transport_matrix = solve_transport(im1_norm, im2_norm, costs)
    return min_cost, transport_matrix

if __name__ == '__main__':
    start_time = time.time()
    image_1_path = r'/images/mnist_image_1.jpg'
    image_1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)

    # image_2_path = r'C:\Users\eriki\Documents\school\Thesis\Optimal_transport_playground\images\im2_test.png'
    # image_moved_left = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
    image_moved_left = image_2_pixels_left(image_1)

    image_1 = image_1.astype(np.float64)
    image_moved_left = image_moved_left.astype(np.float64)
    image_1 = image_1 / sum(sum(image_1))  # Normalize
    image_moved_left = image_moved_left / sum(sum(image_moved_left))  # Normalize

    source_im_1d = image_1.flatten()
    target_im_1d = image_moved_left.flatten()

    costs = calculate_costs(len(image_1.flatten()))

    # min_cost, transport_matrix = solve_transport(source_im_1d, target_im_1d, costs)

    # print(min_cost)
    # print(transport_matrix)
    end = time.time()
    elaplsed = end - start_time
    print(f"elapsed time for my implementation: {elaplsed} seconds")

    start_time_pot = time.time()
    wasserstein_dist = ot.emd2(image_1.flatten(), image_moved_left.flatten(), costs)
    end_time_pot = time.time()
    elapsed = end_time_pot - start_time_pot
    wasserstein_dist_exact = ot.lp.emd(image_1.flatten(), image_moved_left.flatten(), costs)
    elapsed_exact_pot = time.time() - end_time_pot
    print(f"elapsed time for pot implementation: {elapsed} seconds")
    print(f"elapsed time for exact pot implementation: {elapsed_exact_pot} seconds")
