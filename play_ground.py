import numpy as np
import pandas as pd
from utils.Visualizations import *

# Measures:
columns = ['Res', 'Noise_Param', 'Scale_Param', 'Distances_Classic', 'Distances_Noised',
           'Ratios_EMD', 'Distances_Linear', 'Distances_Linear_Noised', 'Ratios_Linear']
df = pd.DataFrame(columns=columns)

# TODO : add original distance as a parameter and column

res_values = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
noise_values = np.logspace(start=-3, stop=1, num=22)  # We want a multiplication of 3 + 1 because we start at 0
scale_values = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
# reg_m_values = [1, 10, 30]  # This is not needed now since we are preforming balanced ot

for res in res_values:
    for noise in noise_values:
        for scale in scale_values:
            df = run_experiment_and_append(df, res=res, noise_param=noise, scale_param=scale)
    print('Done with res: ', res)
df.to_csv('results_measures_expanded.csv', index=False)

# Images:
columns = ['Noise_Param', 'Im_Size', 'Distances_Classic', 'Distances_Noised', 'Ratios_EMD',
           'Distances_Linear', 'Distances_Linear_Noised', 'Ratios_Linear']
df = pd.DataFrame(columns=columns)

# im_sizes = [10, 20, 50, 100]
# noises = np.linspace(start=1e-3, stop=1, num=22)
#
# for im_size in im_sizes:
#     im1 = np.zeros((im_size, im_size))
#     im1[int(0.1 * im_size): int(0.3 * im_size), int(0.1 * im_size): int(0.3 * im_size)] = 1
#
#     im2 = np.zeros((im_size, im_size))
#     im2[int(0.7 * im_size): int(0.9 * im_size), int(0.7 * im_size): int(0.9 * im_size)] = 1
#
#     for noise in noises:
#         df = run_experiment_and_append_images(df, im1, im2, noise_param=noise)
#     print('Done with size: ', im_size)
#
# df.to_csv('results_images.csv', index=False)
