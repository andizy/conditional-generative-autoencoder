from re import L
import torch
import torch.nn.functional as F

from autoencoder import Autoencoder, CondEncoder, CondDecoder
from flow_model import real_nvp, glow

import numpy as np

from my_utils import sampling, flags, get_default_devices, NFType, SNR, SSIM
from datasets import load_dataset, DatasetType
from dataset_stat import *
from pgd import *

import os

import cv2


def delta_value():
    torch.manual_seed(0)
    np.random.seed(0)
    FLAGS, unparsed = flags()
    dataset = FLAGS.dataset
    gpu_num = FLAGS.gpu_num
    desc = FLAGS.desc
    image_size = FLAGS.res
    c = FLAGS.channel
    enable_cuda = True
    device = get_default_devices(gpu_num=gpu_num, enable_cuda=enable_cuda)
    flow_depth = FLAGS.flow_depth
    latent_dim = FLAGS.latent_dim
    batch_size = FLAGS.batch_size
    flow_type = FLAGS.flow_type
    train_delta = bool(FLAGS.train_delta)
    mean_sample = bool(FLAGS.mean_sample)
    
    

    #Experiment path
    all_experiments = 'experiments/'
    assert os.path.exists(all_experiments), "You dont have an experiment"
    # experiment path
    exp_path = all_experiments + 'Autoencoder_' + dataset + '_' \
        + str(flow_depth) + '_' + str(latent_dim) + '_' + str(image_size) + '_' + desc
    assert os.path.exists(exp_path), f"Experiment :'{exp_path}' is not existing, pls check paramethers"
    train_dataset, test_dataset = load_dataset(
        dataset_type=dataset,
        img_size=(image_size, image_size),
        c=c
        )
    y_train_ctrl = False
    if dataset == DatasetType.limited_ct:
        y_train_ctrl = True
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle = True,num_workers=8)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle = True,num_workers=8)
    
    
    # Xtr_mean, Xtr_var = dataset_mean_std(train_loader)
    learning_rate = 1e-4
    step_size = 50
    gamma = 0.5
    #load autoencoder 
    enc = CondEncoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
    dec = CondDecoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
    aeder = Autoencoder(encoder = enc , decoder = dec).to(device)
    
    optimizer_aeder = torch.optim.Adam(aeder.parameters(), lr=learning_rate)
    scheduler_aeder = torch.optim.lr_scheduler.StepLR(optimizer_aeder, step_size=step_size, gamma=gamma)

    
    checkpoint_autoencoder_path = os.path.join(exp_path, 'autoencoder.pt')
    assert os.path.exists(checkpoint_autoencoder_path), f"The aoutoencoder '{str(checkpoint_autoencoder_path)}' is not existing!!"
    checkpoint_autoencoder = torch.load(checkpoint_autoencoder_path)
    aeder.load_state_dict(checkpoint_autoencoder['model_state_dict'])
    optimizer_aeder.load_state_dict(checkpoint_autoencoder['optimizer_state_dict'])
    print('Autoencoder is restored...')
    print('Mean Sample', mean_sample)
    
    #load  NFM

    if flow_type == NFType.real_nvp:
        nfm = real_nvp(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    else:
        nfm = glow(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    with torch.no_grad():
        nfm = nfm.to(device)
        optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
        checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
        assert os.path.exists(checkpoint_flow_path), f"The NFM '{str(checkpoint_autoencoder_path)}' is not existing!!"
        
        checkpoint_flow = torch.load(checkpoint_flow_path)
        nfm.load_state_dict(checkpoint_flow['model_state_dict'])
        optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
        print('Flow model is restored...')
    
    adver_path_generated = os.path.join(
                exp_path, 'adversarial_robustness')
    if os.path.exists(adver_path_generated) == False:
            os.mkdir(adver_path_generated)
    
    if mean_sample:
        gen_img_path_generated = os.path.join(
                    adver_path_generated, 'generated_images_mu')
    else:
        gen_img_path_generated = os.path.join(
                    adver_path_generated, 'generated_images')
    if os.path.exists(gen_img_path_generated) == False:
            os.mkdir(gen_img_path_generated)
    
    first_loop = True
    for i,Ytr in enumerate(train_loader):
        if first_loop:
            if train_delta:
                
                # delta = fgsm_relnvp(nfm=nfm, aem=aeder,
                #             X=Ytr[0], y=Ytr[1], mean_sample=mean_sample,
                #             epsilon=2, alpha=1e-5 , num_iter=100,
                #             device=device, exp_path=adver_path_generated)
                delta = batch_delta(nfm=nfm, aem=aeder,
                            X=Ytr[0], y=Ytr[1],
                            epsilon=4, alpha=0.1 , num_iter=30,
                            device=device)
                print("Delta is computed ... ", delta.shape)
                if mean_sample:
                    torch.save({
                                    'delta': delta,
                                    }, os.path.join(exp_path, 'delta_mu.pt'))
                else:
                    torch.save({
                                    'delta': delta,
                                    }, os.path.join(exp_path, 'delta.pt'))
                print("Training of delta is done ..." )
            else:
                if mean_sample:
                    delta = torch.load(os.path.join(exp_path, 'delta_mu.pt'))['delta']
                else:
                    delta = torch.load(os.path.join(exp_path, 'delta.pt'))['delta']
                print("Loading of deta is done ... ")
            first_loop = False
        
        delta = delta.to(device)
        z_delta, _ = nfm.sample(Ytr[1].to(device)+delta)
        # x_hat_delta = aeder.decoder(z_delta, Ytr[1].to(device))
        x_hat_delta = aeder.decoder(z_delta, Ytr[1].to(device)+delta)
        
        z, _ = nfm.sample(Ytr[1].to(device))
        x_hat = aeder.decoder(z, Ytr[1].to(device))

        x_hat = x_hat.detach().cpu().numpy()
        x_hat_delta = x_hat_delta.detach().cpu().numpy()
        # x_hat = x_hat.detach().cpu().numpy().transpose(0,2,3,1)
        # x_hat_delta = x_hat_delta.detach().cpu().numpy().transpose(0,2,3,1)

        snr_delta = SNR(Ytr[0].numpy(), x_hat_delta,)
        snr = SNR(Ytr[0].numpy(), x_hat,)
        snr_between = SNR( x_hat, x_hat_delta,)
        
        # ssim_delta = 0
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
        
        if i == 30:
            break
        
if __name__ == "__main__":
    delta_value()
