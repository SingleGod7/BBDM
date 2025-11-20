from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import numpy as np
import os
import tqdm
import pandas as pd

def psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=1.0)

metric_list = []

for i in tqdm.tqdm(os.listdir('./results/t12fdg/BrownianBridge/sample_to_eval/ground_truth_np')):
    gt = np.load(os.path.join('./results/t12fdg/BrownianBridge/sample_to_eval/ground_truth_np', i))
    result = np.load(os.path.join('./results/t12fdg/BrownianBridge/sample_to_eval/result_np', i))
    gt = gt.squeeze()
    result = result.squeeze()
    p = psnr(gt, result)
    s = ssim(gt, result)
    tqdm.tqdm.write(f'{i} psnr: {p}, ssim: {s}')
    metric_list.append({'psnr': p, 'ssim': s})

metric_df = pd.DataFrame(metric_list)

print(f'psnr: {np.mean(metric_df["psnr"]):.4f} ± {np.std(metric_df["psnr"]):.4f}')
print(f'ssim: {np.mean(metric_df["ssim"]):.4f} ± {np.std(metric_df["ssim"]):.4f}')

metric_df.to_csv('./results/t12fdg/BrownianBridge/metric.csv', index=False)



