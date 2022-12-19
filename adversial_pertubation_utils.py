
from re import L
import torch
import torch.nn.functional as F

from autoencoder import Autoencoder, CondEncoder, CondDecoder
from flow_model import real_nvp, glow

import numpy as np

from my_utils import sampling, flags, get_default_devices, NFType, SNR, SSIM
from datasets import load_dataset, DatasetType
from dataset_stat import *
from skimage import metrics
import os

import cv2


def add_delta_test_dataset(Ytr, x_hat, x_hat_delta, i, mean_sample, adver_path_generated, gen_img_path_generated):
    
    
    # snr_delta = SNR(Ytr[0].numpy(), x_hat_delta,)
    # snr = SNR(Ytr[0].numpy(), x_hat,)
    # snr_between = SNR( x_hat, x_hat_delta,)
    
    snr_delta = metrics.peak_signal_noise_ratio(Ytr[0].numpy(), x_hat_delta,)
    snr = metrics.peak_signal_noise_ratio(Ytr[0].numpy(), x_hat,)
    snr_between = metrics.peak_signal_noise_ratio( x_hat, x_hat_delta,)
    
    # # ssim_delta = 0
    ssim_delta = SSIM(Ytr[0].numpy(),x_hat_delta )
    ssim = SSIM(Ytr[0].numpy(),x_hat)
    ssim_between = SSIM(x_hat,x_hat_delta)
    
    x_hat = x_hat.transpose(0,2,3,1)
    x_hat_delta = x_hat_delta.transpose(0,2,3,1)
    print(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
    if mean_sample:
        with open(os.path.join(adver_path_generated, 'results_mu.txt'), 'a') as file:
                        file.write(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
                        file.write('\n')
    else:
        with open(os.path.join(adver_path_generated, 'results.txt'), 'a') as file:
                        file.write(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
                        file.write('\n')
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}_delta.png"), x_hat_delta[0]*255)
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}_gen.png"), x_hat[0]*255)
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}.png"), Ytr[0].numpy().transpose(0,2,3,1)[0]*255)

def add_delta_posterio_sample(Ytr, x_hat, x_hat_delta, i, mean_sample, adver_path_generated, gen_img_path_generated):
    
    # snr_delta = SNR(Ytr[0].numpy(), x_hat_delta,)
    # snr = SNR(Ytr[0].numpy(), x_hat,)
    # snr_between = SNR( x_hat, x_hat_delta,)
    snr_delta = metrics.peak_signal_noise_ratio(Ytr[0].numpy(), x_hat_delta,)
    snr = metrics.peak_signal_noise_ratio(Ytr[0].numpy(), x_hat,)
    snr_between = metrics.peak_signal_noise_ratio( x_hat, x_hat_delta,data_range=1)
    # snr_between = 0
    # ssim_delta = 0
    ssim_delta = SSIM(Ytr[0].numpy(),x_hat_delta )
    ssim = SSIM(Ytr[0].numpy(),x_hat)
    ssim_between = SSIM(x_hat,x_hat_delta)
    x_error = np.abs(x_hat - x_hat_delta)
    x_hat = x_hat.transpose(0,2,3,1)
    x_hat_delta = x_hat_delta.transpose(0,2,3,1)
    x_error = x_error.transpose(0,2,3,1)
    print(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
    if mean_sample:
        with open(os.path.join(adver_path_generated, 'results_mu.txt'), 'a') as file:
                        file.write(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
                        file.write('\n')
    else:
        with open(os.path.join(adver_path_generated, 'results.txt'), 'a') as file:
                        file.write(f"img:{i}, SNR Delta: {snr_delta}, SNR: {snr}, SNR between: {snr_between}, SSIM Delta: {ssim_delta}, SSIM: {ssim}, SSIM between: {ssim_between}")
                        file.write('\n')
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}_delta.png"), x_hat_delta[0]*255)
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}_gen.png"), x_hat[0]*255)
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}_error.png"), x_error[0]*255)
    cv2.imwrite(os.path.join(gen_img_path_generated, f"img_{i}.png"), Ytr[0].numpy().transpose(0,2,3,1)[0]*255)
        