import numpy as np
import pandas as pd
from utils.Visualizations import *
import time

df = pd.DataFrame()
res_values = [int(x) for x in np.linspace(start=100, stop=300, num=3)]
# res_values = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
SNR_values = np.logspace(start=-2, stop=1, num=31)  # We want a multiplication of 3 + 1 because we start at 0
scale = 1
for res in res_values:
    for SNR in SNR_values:
        X = np.linspace(0, scale, res)
        sample_distribution = norm.pdf(X, scale * 0.65, scale * 0.1)
        sample_distribution = sample_distribution / sample_distribution.sum()
        signal_power = (sample_distribution ** 2).sum()
        noise = noise_from_SNR(SNR, signal_power=signal_power, res=res)
        df = run_experiment_and_append(df, res=res, noise_param=noise, scale_param=scale, SNR=SNR, num_samples=200)
    print('Done with res: ', res)
# df.to_csv('results_measures_expanded.csv', index=False)
df.to_csv('csvs/results_measures_SNR_test.csv', index=False)

# Images:
# columns = ['Noise_Param', 'Im_Size', 'Distances_Classic', 'Distances_Noised', 'Ratios_EMD',
#            'Distances_Linear', 'Distances_Linear_Noised', 'Ratios_Linear']
# df = pd.DataFrame(columns=columns)

# im_size_values = [int(x) for x in np.linspace(start=10, stop=50, num=5)]
# SNR_values = np.logspace(start=3, stop=-2, num=31)
# df = pd.DataFrame()
#
# for im_size in im_size_values:
#     start_time = time.time()
#     im1 = np.zeros((im_size, im_size))
#     im1[int(0.1 * im_size): int(0.3 * im_size), int(0.1 * im_size): int(0.3 * im_size)] = 1
#     im2 = np.zeros((im_size, im_size))
#     im2[int(0.7 * im_size): int(0.9 * im_size), int(0.7 * im_size): int(0.9 * im_size)] = 1
#
#     for SNR in SNR_values:
#         df = run_experiment_and_append_images(df, im1, im2, SNR=SNR)
#
#     print(f'Done with images of size {im_size},  it took {time.time() - start_time} seconds')
# df.to_csv('csvs/results_images_3_12.csv', index=False)
