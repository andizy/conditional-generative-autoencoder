

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchdata.datapipes.map import SequenceWrapper

import numpy as np
import shutil
import os

from autoencoder import Autoencoder, Encoder, Decoder, CondEncoder, CondDecoder
from conditional_network import CondNetNF
from flow_model import real_nvp, glow

from my_utils import *
from datasets import *
from fit import fit_aeder, fit_flow, fit_flow_with_delta
from logger_conf import logger

from timeit import default_timer



    
def train():
    torch.manual_seed(0)
    np.random.seed(0)
    FLAGS, unparsed = flags()
    epochs_flow = FLAGS.epochs_flow
    epochs_aeder = FLAGS.epochs_aeder
    flow_depth = FLAGS.flow_depth
    latent_dim = FLAGS.latent_dim
    batch_size = FLAGS.batch_size
    dataset = FLAGS.dataset
    gpu_num = FLAGS.gpu_num
    desc = FLAGS.desc
    image_size = FLAGS.res
    c = FLAGS.channel
    flow_type = FLAGS.flow_type
    remove_all = bool(FLAGS.remove_all)
    train_aeder = bool(FLAGS.train_aeder)
    train_flow = bool(FLAGS.train_flow)
    restore_flow = bool(FLAGS.restore_flow)
    add_delta = bool(FLAGS.add_delta)
    enable_cuda = True
    device = get_default_devices(gpu_num=gpu_num, enable_cuda=enable_cuda)
    all_experiments = 'experiments/'
    if os.path.exists(all_experiments) == False:
        os.mkdir(all_experiments)
    # experiment path
    exp_path = all_experiments + 'Autoencoder_' + dataset + '_' \
        + str(flow_depth) + '_' + str(latent_dim) + '_' + str(image_size) + '_' + desc

    if os.path.exists(exp_path) == True and remove_all == True:
        shutil.rmtree(exp_path)

    if os.path.exists(exp_path) == False:
        os.mkdir(exp_path)

    #Dataset
    train_dataset, test_dataset = load_dataset(
        dataset_type=dataset,
        img_size=(image_size, image_size),
        c=c
        )
    y_train_ctrl = False
    if dataset == DatasetType.limited_ct:
        y_train_ctrl = True
        
        
        

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle = True,num_workers=8)
    
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle = True,num_workers=8)
    



    ntrain = len(train_loader.dataset)

  
    learning_rate = 1e-4
    step_size = 50
    gamma = 0.5
    myloss = F.mse_loss
    plot_per_num_epoch = 1 if ntrain > 20000 else 20000//ntrain

    # Print the experiment setup:
    print('Experiment setup:')
    print('---> epochs_flow: {}'.format(epochs_flow))
    print('---> flow_depth: {}'.format(flow_depth))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(learning_rate))
    print('---> experiment path: {}'.format(exp_path))
    print('---> latent dim: {}'.format(latent_dim))
    print('---> image size: {}'.format(image_size))
    print('---> Number of training samples: {}'.format(ntrain))
    

    logger.info("Training phase ...")
    # 1. Training Autoencoder:
    #specify the Autoencode model
    enc = CondEncoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
    dec = CondDecoder(latent_dim = latent_dim, in_res = image_size , c = c).to(device)
    aeder = Autoencoder(encoder = enc , decoder = dec).to(device)
    num_param_aeder= count_parameters(aeder)
    print('---> Number of trainable parameters of Autoencoder: {}'.format(num_param_aeder))

    optimizer_aeder = Adam(aeder.parameters(), lr=learning_rate)
    scheduler_aeder = torch.optim.lr_scheduler.StepLR(optimizer_aeder, step_size=step_size, gamma=gamma)

    checkpoint_autoencoder_path = os.path.join(exp_path, 'autoencoder.pt')
    if os.path.exists(checkpoint_autoencoder_path):
        checkpoint_autoencoder = torch.load(checkpoint_autoencoder_path)
        aeder.load_state_dict(checkpoint_autoencoder['model_state_dict'])
        optimizer_aeder.load_state_dict(checkpoint_autoencoder['optimizer_state_dict'])
        print('Autoencoder is restored...')

    if train_aeder:
        fit_aeder(aeder,
                myloss,
                optimizer_aeder, 
                scheduler_aeder, 
                plot_per_num_epoch, 
                epochs_aeder, 
                train_loader,
                device,
                #additional atttributes
                ntrain,
                exp_path,
                checkpoint_autoencoder_path,
                image_size,
                c,
                y_train_loader=y_train_ctrl if y_train_ctrl else None                
                )

    #2.Intilize the nfm model
    if flow_type == NFType.real_nvp:
        nfm = real_nvp(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)
    else:
        nfm = glow(latent_dim = latent_dim, K = flow_depth, in_res = image_size , c = c)

    nfm = nfm.to(device)
    num_param_nfm = count_parameters(nfm)
    print('Number of trainable parametrs of flow: {}'.format(num_param_nfm))
    optimizer_flow = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=step_size, gamma=gamma)
    
    # Initialize ActNorm
    if y_train_ctrl:
        batch_img_touple = next(iter(train_loader))
        batch_img = batch_img_touple[0].to(device)
        cond_batch_img = batch_img_touple[1].to(device)
    else:
        batch_img = next(iter(train_loader)).to(device)
        cond_batch_img = add_noise(batch_img)

    
    dummy_samples = aeder.encoder(batch_img, cond_batch_img)
    dummy_samples = dummy_samples.view(-1, latent_dim)
    cond_dummy_samples = add_noise(dummy_samples)
    likelihood = nfm.log_prob(dummy_samples, cond_batch_img)
    
    checkpoint_flow_path = os.path.join(exp_path, 'flow.pt')
    if os.path.exists(checkpoint_flow_path) and restore_flow == True:
        checkpoint_flow = torch.load(checkpoint_flow_path)
        nfm.load_state_dict(checkpoint_flow['model_state_dict'])
        optimizer_flow.load_state_dict(checkpoint_flow['optimizer_state_dict'])
        print('Flow model is restored...')
    

    if train_flow:
        if add_delta:
            fit_flow_with_delta(nfm, 
                aeder,
                optimizer_flow,
                scheduler_flow,
                epochs_flow, 
                train_loader,
                device,
                plot_per_num_epoch,
                ntrain,
                exp_path,
                checkpoint_flow_path,
                image_size,
                c,
                y_train_loader=y_train_ctrl
                )
        else:
            fit_flow(nfm, 
                aeder,
                optimizer_flow,
                scheduler_flow,
                epochs_flow, 
                train_loader,
                device,
                plot_per_num_epoch,
                ntrain,
                exp_path,
                checkpoint_flow_path,
                image_size,
                c,
                y_train_loader=y_train_ctrl
                )

    image_path_generated = os.path.join(
                exp_path, 'test_cond_generated')
    
    
    if os.path.exists(image_path_generated) == False:
            os.mkdir(image_path_generated)
    
    n_test = 5 # Number of test samples
    n_sample_show = 4 # Number of posterior samples to show for each test sample
    n_average = 25 # number of posterior samples used for MMSE and UQ estimation
    
    for i,y in enumerate(test_loader):
        x_sampled_conditional = conditional_sampling(nfm,aeder, 
                                                    y[0],y[1],n_average,
                                                    n_test, n_sample_show, 
                                                    device)[0]
        
        cv2.imwrite(os.path.join(image_path_generated, f'posterior_samples_{i}.png'),
                    x_sampled_conditional[:, :, :, ::-1].reshape(
            n_test, n_sample_show + 5,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(n_test*image_size, -1, c)*127.5 + 127.5)

if __name__ == '__main__':
    train()