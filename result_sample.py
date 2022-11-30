import torch
import torch.nn.functional as F

from autoencoder import Autoencoder, CondEncoder, CondDecoder
from flow_model import real_nvp, glow

import numpy as np

from my_utils import sampling, flags, get_default_devices, NFType, conditional_sampling
from datasets import load_dataset, DatasetType
from dataset_stat import *

import os

import cv2


def cond_sample():
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

    if flow_type == NFType.real_nvp:
        nfm = real_nvp(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    else:
        nfm = glow(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    nfm = nfm.to(device)
    optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
    checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
    assert os.path.exists(checkpoint_flow_path), f"The NFM '{str(checkpoint_autoencoder_path)}' is not existing!!"
    
    checkpoint_flow = torch.load(checkpoint_flow_path)
    nfm.load_state_dict(checkpoint_flow['model_state_dict'])
    optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
    print('Flow model is restored...')
    
    
    image_path_generated = os.path.join(
                exp_path, 'cond_generated')
    samples_name="generated_sample"
    if os.path.exists(image_path_generated) == False:
            os.mkdir(image_path_generated)
    
    y = next(iter(test_loader))

    # samples = sampling(y, aeder, nfm, device=device)
    #conditional sampling from posterio distribution
    

    n_test = 5 # Number of test samples
    n_sample_show = 4 # Number of posterior samples to show for each test sample
    n_average = 25 # number of posterior samples used for MMSE and UQ estimation
    print('Start conditional sampling...')
    for i,y in enumerate(test_loader):
        x_sampled_conditional = conditional_sampling(nfm,aeder, 
                                y[0],y[1],n_average,
                                n_test, n_sample_show, 
                                device, exp_path=exp_path)[0]
        cv2.imwrite(os.path.join(image_path_generated, f'posterior_samples_{i}.png'),
                        x_sampled_conditional[:, :, :, ::-1].reshape(
                n_test, n_sample_show + 5,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(n_test*image_size, -1, c)*255)
    
if __name__ == "__main__":
    with torch.no_grad():
        cond_sample()

