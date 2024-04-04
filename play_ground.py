import os
import cv2
import glob
import itertools
from utils.utils import *
from utils.image import Image

dotmark_pictures_path = "..\\DOTmark_1.0\\Pictures\\"
full_path = os.path.join(os.getcwd(), dotmark_pictures_path)
resolutions = [32, 64, 128, 256]
# resolutions = [32]
image_numbers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
SNR_values = np.logspace(start=5, stop=1, num=11)
noise_values = np.logspace(start=-3, stop=1, num=11)
df_im_l1 = pd.DataFrame()

categories_pattern = os.path.join(dotmark_pictures_path, "*")
category_dirs = [path for path in glob.glob(categories_pattern) if os.path.isdir(path)]
category_names = [os.path.basename(path) for path in category_dirs]

category_names = category_names[:1]

num_samples = 20
pairs = list(itertools.combinations(image_numbers, 2))

df_res = pd.DataFrame()
for category in category_names:
    category_dir = os.path.join(full_path, category)
    print(f"Processing category: {category}")
    for resolution in resolutions:
        print(f"Processing resolution: {resolution}")
        for noise_param in noise_values:
            noise_param_original = noise_param
            noise_param = noise_param / (resolution**2)
            SNR = 1/noise_param
        #for SNR in SNR_values:
        #    noise_param = noise_from_SNR(SNR, 1, resolution)
            for image_pair in pairs:
                image1 = Image(resolution, category, image_pair[0], full_path)
                image2 = Image(resolution, category, image_pair[1], full_path)
                
                f_dist_original, f_time_original = calculate_and_time_fourier1(image1.image, image2.image)
                l2_dist_original, l2_time_original = calculate_and_time_l2(image1.image, image2.image)

                results = Image.analyze_image_pair_without_wasserstein(image1, image2, 
                                                                       num_samples, noise_param)

                f_dist_noised, l2_dist_noised, time_f, time_l2 = results
            
                new_row = {
                    'Category': category,
                    'image1_index': image_pair[0],
                    'image2_index': image_pair[1],
                    'Noise': noise_param_original,
                    'SNR': SNR,
                    'Resolution': resolution,
                    'Fourier Original': f_dist_original, 
                    'Fourier Noised': f_dist_noised,
                    'Fourier Ratio': f_dist_original / f_dist_noised,
                    'Fourier Time': time_f, 
                    'L2 Original': l2_dist_original,
                    'L2 Noised': l2_dist_noised,
                    'L2 Ratio': l2_dist_original / l2_dist_noised,
                    'L2 Time': time_l2}
                df_res = df_res._append(new_row, ignore_index=True)

df_res.to_csv("results2.csv", index=False)