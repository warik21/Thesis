import os
import cv2
import glob
import itertools
import sys

utils_path = os.path.abspath(r"C:\Users\eriki\OneDrive\Documents\all_folder\Thesis\Thesis")
if utils_path not in sys.path:
    sys.path.append(utils_path)
from utils.utils import *

dotmark_pictures_path = "..\\DOTmark_1.0\\Pictures\\"
full_path = os.path.join(os.getcwd(), dotmark_pictures_path)
# resolutions = [32, 64, 128, 256, 512]
resolutions = [32]
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
SNR_values = np.logspace(start=3, stop=-2, num=31)
df_im_l1 = pd.DataFrame()

# Define the pattern to match all items in the directory
categories_pattern = os.path.join(dotmark_pictures_path, "*")
# Use a list comprehension to filter only directories
category_dirs = [path for path in glob.glob(categories_pattern) if os.path.isdir(path)]
# Extract just the category names from the full paths
category_names = [os.path.basename(path) for path in category_dirs]

pairs = list(itertools.combinations(image_numbers, 2))

for category in category_names:
    category_dir = os.path.join(full_path, category)
    print(f"Processing category: {category}")
    for resolution in resolutions:
        for SNR in SNR_values:
            for image_pair in pairs:
                # Here we would like to noise and compare each pair of images. We would want to create a confusion matrix
                # We essentially want this step to output 3 things:
                # Conf_mat(I1_noised,I2_noised)
                # Conf_mat(I1, I2)
                # emd(I, I_tilde) for every image, only thing to think about is whether we want to noise at every stage or not.
                path_im1 = os.path.join(category_dir, f"picture{resolution}_10{image_pair[0]}.png")
                im1 = cv2.imread(path_im1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                im1 = cv2.resize(im1, (resolution, resolution))
                path_im2 = os.path.join(category_dir, f"picture{resolution}_10{image_pair[1]}.png")
                im2 = cv2.imread(path_im1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                im2 = cv2.resize(im2, (resolution, resolution))

                df_im_l1 = run_experiment_and_append_images(df=df_im_l1, im1=im1, im2=im2, SNR=SNR,
                                                            distance_metric='L2', n_samples=2)
print('Done')
