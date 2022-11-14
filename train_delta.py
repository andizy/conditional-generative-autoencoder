from re import L
import torch
import torch.nn.functional as F

from autoencoder import Autoencoder, CondEncoder, CondDecoder
from flow_model import real_nvp

import numpy as np

from my_utils import sampling, flags, get_default_devices
from datasets import load_dataset, DatasetType
from dataset_stat import *
from pgd import pgd, pgd_l2, pgd_l2_z_noise, pgd_l2_dataset

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
    myloss = F.mse_loss
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
    
    
    #load  NFM

    nfm = real_nvp(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    nfm = nfm.to(device)
    optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
    checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
    assert os.path.exists(checkpoint_flow_path), f"The NFM '{str(checkpoint_autoencoder_path)}' is not existing!!"
    
    checkpoint_flow = torch.load(checkpoint_flow_path)
    nfm.load_state_dict(checkpoint_flow['model_state_dict'])
    optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
    print('Flow model is restored...')
    Ytr = next(iter(train_loader))
    delta = pgd_l2(nf_model=nfm, ae_model=aeder, Y=Ytr, epsilon=2, alpha=0.1, num_iter=40, device=device)
    # delta = pgd_l2_dataset(nf_model=nfm, ae_model=aeder, Ytr=train_loader, epsilon=2, alpha=0.1, num_iter=40, device=device)
    # delta = pgd_l2_z_noise(nf_model=nfm, ae_model=aeder,latent_space=latent_dim ,Y=Ytr, epsilon=2, alpha=0.1, num_iter=40, device=device)
    
    # delta = pgd(nf_model=nfm, ae_model=aeder,loss=delta_loss, Y=Ytr, epsilon=2, alpha=0.1, num_iter=40, device=device)
    print(delta.shape)
    print(delta.max())
if __name__ == "__main__":
    delta_value()
