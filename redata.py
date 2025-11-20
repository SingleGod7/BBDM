import pandas as pd
import numpy as np
import os
dot_data = pd.read_csv('/data/jiangyizhou/renji/dota.csv')

def normalize_array(array, window_center, window_width, rescale_slope, rescale_intercept):
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    rescaled_array = array * rescale_slope + rescale_intercept
    normalized_array = np.clip(rescaled_array, window_min, window_max)
    normalized_array = (normalized_array - window_min) / (window_max - window_min) # 归一化到 0-1
    return normalized_array

data_dir = "/data/jiangyizhou/renji/t12dota_2d"

for file in os.listdir(os.path.join(data_dir, "test", "dota")):
    if os.path.isdir(os.path.join(data_dir, "test", "dota", file)):
        for j in os.listdir(os.path.join(data_dir, "test", "dota", file)):
            if j.endswith(".npy"):
                array = np.load(os.path.join(data_dir, "test", "dota", file, j))
                window_center = dot_data.loc[dot_data['patient_id'] == file, 'WindowCenter'].values[0]
                window_width = dot_data.loc[dot_data['patient_id'] == file, 'WindowWidth'].values[0]
                rescale_slope = dot_data.loc[dot_data['patient_id'] == file, 'RescaleSlope'].values[0]
                rescale_intercept = dot_data.loc[dot_data['patient_id'] == file, 'RescaleIntercept'].values[0]
                normalized_array = normalize_array(array, window_center, window_width, rescale_slope, rescale_intercept)
                np.save(os.path.join(data_dir, "test", "dota", file, j), normalized_array)
                # if np.max(array) > 1.0 or np.min(array) < 0.0:
                #     print(file, j)