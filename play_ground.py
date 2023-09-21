import pandas as pd
from utils.Visualizations import *

# Initialize DataFrame
columns = ['Res', 'Noise_Param', 'Scale_Param', 'Distances_Classic', 'Distances_Noised',
           'Ratios_EMD', 'Distances_Linear', 'Distances_Linear_Noised', 'Ratios_Linear']
df = pd.DataFrame(columns=columns)

res_values = [10, 50, 100]
noise_values = [5e-2, 1e-2, 5e-3, 1e-3]
scale_values = [1, 10, 20, 100]
reg_m_values = [1, 10, 30]

for res in res_values:
    for noise in noise_values:
        for scale in scale_values:
            for reg_m in reg_m_values:
                df = run_experiment_and_append(df, res=res, noise_param=noise, scale_param=scale, reg_m_param=reg_m)

df.to_csv('results2.csv', index=False)
