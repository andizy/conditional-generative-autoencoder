import argparse
from cmath import sqrt
import numpy as np
import torch
from conditional_network import CondNetNF
import cv2
import torch
import os 

def data_normalization(x):
    '''Normalize data between -1 and 1'''
    x = x.astype('float32')
    x = x - (x.max() + x.min())/2
    x /= (x.max())
    
    return x


def conditional_sampling(pz , inj_model , bij_model , x_test , y_test ,
                         n_average , n_test = 5 , n_sample_show = 4):
    '''Generate posterior samples, MMSE, MAP and UQ'''
    
    def normalization(image , min_x , max_x):
        image += -image.min()
        image /= image.max()
        image *= max_x - min_x
        image += min_x
        
        return image
    
    y_s_single = y_test[1*n_test:2 * n_test]
    y_reshaped = y_test[1*n_test:2 * n_test].numpy()
    if np.shape(y_s_single)[1] != np.shape(x_test)[1]:
        
        r = np.shape(x_test)[1]
        if np.shape(y_test)[3] == 2:
            y_reshaped = np.sqrt(np.sum(np.square(y_reshaped) , axis = 3 , keepdims=True))
            y_reshaped = data_normalization(y_reshaped)
        
        y_reshaped_orig = np.zeros([n_test , np.shape(x_test)[1] , np.shape(x_test)[2] , np.shape(x_test)[3]])
        for i in range(n_test):
            if np.shape(x_test)[3] == 1:
                y_reshaped_orig[i,:,:,0] = cv2.resize(y_reshaped[i][:,:,0] , (r,r),
                                                  interpolation = cv2.INTER_NEAREST)
            else:
                y_reshaped_orig[i] = cv2.resize(y_reshaped[i] , (r,r),
                                            interpolation = cv2.INTER_NEAREST)
         
        y_reshaped = y_reshaped_orig
    
    y_s = tf.repeat(y_s_single, n_average, axis = 0)
    
    gt = x_test[1*n_test:2 * n_test].numpy()
    
    z_random_base = pz.prior.sample(n_average * n_test)
    z_random_base_mean = (z_random_base[:n_test] - pz.mu) * 0 + pz.mu
    
    z_random = bij_model(z_random_base ,
                            y_s,
                            reverse = True)[0] # Intermediate samples
    
    z_random_mean = bij_model(z_random_base_mean ,
                            y_s_single,
                            reverse = True)[0] # Intermediate samples
    
    
    x_sampled = inj_model(z_random,
                      y_s,
                      reverse = True)[0].numpy() 
    
    x_MAP = inj_model(z_random_mean,
                      y_s_single,
                      reverse = True)[0].numpy() 
        
    
    n_sample = n_sample_show + 5 
    final_shape = [n_test*(n_sample), np.shape(x_sampled)[1] , np.shape(x_sampled)[2],
                   np.shape(x_sampled)[3]]
    x_sampled_all = np.zeros(final_shape)
    mean_vec = np.zeros([n_test , np.shape(x_sampled)[1] , np.shape(x_sampled)[2],
                         np.shape(x_sampled)[3]] , dtype = np.float32)
    
    SSIM_MMSE = 0
    SSIM_pseudo = 0
    SSIM_MAP = 0
    for i in range(n_test):
        x_sampled_all[i*n_sample] = gt[i]
        x_sampled_all[i*n_sample + 1] = x_MAP[i]
        x_sampled_all[i*n_sample + 2] = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        x_sampled_all[i*n_sample+3:i*n_sample + 3 + n_sample_show] = x_sampled[i*n_average:i*n_average + n_sample_show]
        x_sampled_all[i*n_sample + 4 + n_sample_show] = y_reshaped[i]
        x_sampled_all[i*n_sample + 3 + n_sample_show] = normalization(np.std(x_sampled[i*n_average:i*n_average + n_average] , axis = 0) , gt.min() , gt.max())

        mean_vec[i] = np.mean(x_sampled[i*n_average:i*n_average + n_average] , axis = 0)
        SSIM_MMSE = SSIM_MMSE + ssim(mean_vec[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
        
        SSIM_pseudo = SSIM_pseudo + ssim(y_reshaped[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
        
        SSIM_MAP = SSIM_MAP + ssim(x_MAP[i] + 1.0,
                           gt[i] + 1.0,
                           data_range=gt.max() - gt.min(),
                           multichannel=True)
    
    snr_pseudo = SNR(gt +1.0 , y_reshaped + 1.0 )
    snr_MMSE = SNR(gt + 1.0 , mean_vec + 1.0)
    snr_MAP = SNR(gt + 1.0 , x_MAP + 1.0)

    print('SNR of pseudo inverse:{:.3f}'.format(snr_pseudo))
    print('SNR of MMSE:{:.3f}'.format(snr_MMSE))
    print('SNR of MAP:{:.3f}'.format(snr_MAP))
    print('SSIM of pseudo inverse:{:.3f}'.format(SSIM_pseudo/n_test))
    print('SSIM of MMSE:{:.3f}'.format(SSIM_MMSE/n_test))
    print('SSIM of MAP:{:.3f}'.format(SSIM_MAP/n_test))
    return x_sampled_all , y_s_single.numpy() , snr_MMSE,  SSIM_MMSE/n_test

def sampling(y_dataset, ae_model, nf_model, device="cpu"):
    """This function can be used to generate sample from the trained model
    """
    samples = []
    for y in y_dataset:
        y = y.to(device)
        sample_num = torch.tensor(y.shape[0]).to(device)
        z_hat = nf_model.sample(y=y, num_samples=sample_num)
        x_hat  = ae_model.decoder(z_hat[0], y).detach().cpu().numpy()
        samples.append((y,x_hat))
    return samples


def SNR(x_true , x_pred):
    '''Calculate SNR of a barch of true and their estimations'''
    x_true = np.reshape(x_true , [np.shape(x_true)[0] , -1])
    x_pred = np.reshape(x_pred , [np.shape(x_pred)[0] , -1])
    
    Noise = x_true - x_pred
    Noise_power = np.sum(np.square(np.abs(Noise)), axis = -1)
    Signal_power = np.sum(np.square(np.abs(x_true)) , axis = -1)
    SNR = 10*np.log10(np.mean(Signal_power/Noise_power))
    return SNR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_mse_loss(x_true, x_pred):
    noise_power = np.sqrt(np.sum(np.square(x_true - x_pred) , axis = [1,2,3]))
    signal_power = np.sqrt(np.sum(np.square(x_true) , axis = [1,2,3]))
    
    return np.mean(noise_power/signal_power)


def add_noise(x):
    """Add noise to the input signal return signal with the same dimensions as the input """
    noise = torch.rand_like(x)
    return x + noise

def flags():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs_aeder',
        type=int,
        default=200,
        help='number of epochs to train autoencoder network')
     
    
    parser.add_argument(
        '--epochs_flow',
        type=int,
        default=300,
        help='number of epochs to train flow network')


    parser.add_argument(
        '--flow_depth',
        type=int,
        default=5,
        help='Number of blocks in flow model')
    
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch_size')
    
    
    parser.add_argument(
        '--dataset', 
        type=str,
        default='mnist',
        help='which dataset to work with')
    
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=16,
        help='latent dimension')
    
    
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=1,
        help='GPU number')

    parser.add_argument(
        '--remove_all',
        type= int,
        default= 1,
        help='Remove the privious experiment if exists')


    parser.add_argument(
        '--desc',
        type=str,
        default='Default',
        help='add a small descriptor to autoencoder experiment')


    parser.add_argument(
        '--res',
        type=int,
        default=64,
        help='Resolution of the dataset')
    
    parser.add_argument(
        '--channel',
        type=int,
        default=1,
        help='Channel of the dataset')
    
    
    parser.add_argument(
        '--train_aeder',
        type=int,
        default=1,
        help='Train autoencoder network')


    parser.add_argument(
        '--train_flow',
        type=int,
        default=1,
        help='Train normalizing flow network')
    
    parser.add_argument(
        '--restore_flow',
        type=int,
        default=1,
        help='Restore the trained flow if exists')
    
    parser.add_argument(
        '--test_pct',
        type=int,
        default=1,
        help='Percentage to split the test dataset')
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

def get_default_devices(gpu_num=0,enable_cuda=True):
    device = torch.device("cpu")
    if torch.cuda.is_available() and enable_cuda:
        device = torch.device('cuda:' + str(gpu_num))
    return device

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def mean(x):
    """Compute the mean of a dataset"""

    