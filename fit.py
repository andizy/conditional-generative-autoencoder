


import torch
from torch.optim import Adam

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from timeit import default_timer

import normflow as nf

from my_utils import *
from datasets import *
from pgd import *

from logger_conf import logger

from tqdm import tqdm




torch.manual_seed(0)
np.random.seed(0)


def fit_aeder(  aeder,
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
                y_train_loader=None               
                ):
    print("Train AoutEncoder ")
    if plot_per_num_epoch == -1:
        plot_per_num_epoch = epochs_aeder + 1 # only plot in the last epoch
    
    loss_ae_plot = np.zeros([epochs_aeder])
    for ep in range(epochs_aeder):
        aeder.train()
        t1 = default_timer()
        loss_ae_epoch = 0

        for _,image in tqdm(enumerate(train_loader)):
            
            if y_train_loader:
                cond_image = image[1].to(device)
                image = image[0]
            else:
                cond_image = add_noise(image).to(device)
            image = image.to(device)
            batch_size = image.shape[0]
            if image.shape[0] != cond_image.shape[0]:
                cond_image = cond_image[0:image.shape[0]]
            optimizer_aeder.zero_grad()
            embed = aeder.encoder(image, cond_image)
            image_recon = aeder.decoder(embed, cond_image)

            recon_loss = myloss(image_recon.reshape(batch_size, -1) , image.reshape(batch_size, -1) )
            regularization = myloss(embed, torch.zeros(embed.shape).to(device))
            ae_loss = recon_loss + regularization

            ae_loss.backward()
    
            optimizer_aeder.step()
            loss_ae_epoch += ae_loss.item()
            
        scheduler_aeder.step()

        t2 = default_timer()

        loss_ae_epoch/= ntrain
        loss_ae_plot[ep] = loss_ae_epoch
        
        plt.plot(np.arange(epochs_aeder)[:ep], loss_ae_plot[:ep], 'o-', linewidth=2)
        plt.title('AE_loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE loss')

        plt.savefig(os.path.join(exp_path, 'Autoencoder_loss.jpg'))
        np.save(os.path.join(exp_path, 'Autoencoder_loss.npy'), loss_ae_plot[:ep])
        plt.close()
        
        torch.save({
                    'model_state_dict': aeder.state_dict(),
                    'optimizer_state_dict': optimizer_aeder.state_dict()
                    }, checkpoint_autoencoder_path)


        samples_folder = os.path.join(exp_path, 'Generated_samples')
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions_aeder')
    
        if not os.path.exists(image_path_reconstructions):
            os.mkdir(image_path_reconstructions)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_aeder:
            
            sample_number = 9

            ngrid = int(np.sqrt(sample_number))
            
            image_np = image.detach().cpu().numpy().transpose(0,2,3,1)

            image_mat_write = image_np[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size,c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0

            cv2.imwrite(os.path.join(image_path_reconstructions, 'gt_epoch %d.png' % (ep,)),
                        image_mat_write) # Reconstructed training images
            
            
            #writing the image with noise 
            cond_img_np = cond_image.detach().cpu().numpy().transpose(0,2,3,1)
            cond_img_mat_write = cond_img_np[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size,c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0
            cv2.imwrite(os.path.join(image_path_reconstructions, 'noise_image %d.png' % (ep,)),
                        cond_img_mat_write) # Reconstructed training images
            
            
            embed = aeder.encoder(image, cond_image)
            image_recon = aeder.decoder(embed, cond_image)
            image_recon_out = image_recon.detach().cpu().numpy().transpose(0,2,3,1)
            image_recon_write = image_recon_out[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0

            cv2.imwrite(os.path.join(image_path_reconstructions, 'recon%d.png' % (ep,)),
                            image_recon_write) # mesh_based_recon
            
            snr_aeder = SNR(image_np , image_recon_out)


            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                        file.write('ep: %03d/%03d | time: %.4f | aeder_loss %.4f | SNR_aeder  %.4f' %(ep, epochs_aeder,t2-t1,
                            loss_ae_epoch, snr_aeder))
                        file.write('\n')

            print('ep: %03d/%03d | time: %.4f | aeder_loss %.4f | SNR_aeder  %.4f' %(ep, epochs_aeder,t2-t1,
                            loss_ae_epoch, snr_aeder))
        


def fit_flow(nfm, 
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
            y_train_loader=None
            ):
    
    loss_hist = np.array([])

    for ep in range(epochs_flow):

        nfm.train()
        t1 = default_timer()
        loss_flow_epoch = 0
        for _, image in tqdm(enumerate(train_loader)):
            optimizer_flow.zero_grad()
            #add noise to the image to use is as conditional image
            if y_train_loader:
                cond_image = image[1].to(device)
                image = image[0]
            else:
                cond_image = add_noise(image).to(device)
            image = image.to(device)
            if image.shape[0] != cond_image.shape[0]:
                cond_image = cond_image[0:image.shape[0]]
            
            
            y = cond_image 
            x = aeder.encoder(image, cond_image)
            # Compute loss
            loss_flow = nfm.forward_kld(x, y)
            
            if ~(torch.isnan(loss_flow) | torch.isinf(loss_flow)):
                loss_flow.backward()
                optimizer_flow.step()
            
            # Make layers Lipschitz continuous
            nf.utils.update_lipschitz(nfm, 5)
            
            loss_flow_epoch += loss_flow.item()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss_flow.to('cpu').data.numpy())
        
        scheduler_flow.step()
        t2 = default_timer()
        loss_flow_epoch /= ntrain
        
        torch.save({
                    'model_state_dict': nfm.state_dict(),
                    'optimizer_state_dict': optimizer_flow.state_dict()
                    }, checkpoint_flow_path)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_flow:
            samples_folder = os.path.join(exp_path, 'Generated_samples')
            if not os.path.exists(samples_folder):
                os.mkdir(samples_folder)
            image_path_generated = os.path.join(
                samples_folder, 'generated')
        
            if not os.path.exists(image_path_generated):
                os.mkdir(image_path_generated)
            sample_number = 9
            ngrid = int(np.sqrt(sample_number))
            
            # #####Generating sample data########
            num_samples = torch.tensor(sample_number).to(device)
            y_copy = y[0:sample_number,]
            cond_image_bound = cond_image[0:sample_number]
            y_copy = y_copy.to(device)
            cond_image_bound = cond_image_bound.to(device)
            generated_embed, _ = nfm.sample(y=y_copy, num_samples=num_samples)
            generated_embed_mu, _ = nfm.sample_mu(y=y_copy, num_samples=num_samples)
            
            generated_samples = aeder.decoder(generated_embed, cond_image_bound).detach().cpu().numpy()
            generated_samples_mu = aeder.decoder(generated_embed_mu, cond_image_bound).detach().cpu().numpy()
            
            generated_samples = np.reshape(generated_samples,
                                           [generated_samples.shape[0],
                                            c,image_size, image_size]).transpose(0,2,3,1)
            generated_samples_mu = np.reshape(generated_samples_mu,
                                           [generated_samples_mu.shape[0],
                                            c,image_size, image_size]).transpose(0,2,3,1)
            gt = np.reshape(image.detach().cpu().numpy(),
                                           [image.shape[0],
                                            c,image_size, image_size]).transpose(0,2,3,1)

            generated_samples = generated_samples[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0
            
            generated_samples_mu = generated_samples_mu[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0
            
            gt = gt[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0

     
            cv2.imwrite(os.path.join(image_path_generated, 'epoch%d.png' % (ep,)), generated_samples) 
            cv2.imwrite(os.path.join(image_path_generated, 'epoch_mu_%d.png' % (ep,)), generated_samples) 
            cv2.imwrite(os.path.join(image_path_generated, 'epoch_gt_%d.png' % (ep,)), gt) 

            
            
             
            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                    file.write('ep: %03d/%03d | time: %.4f | ML_loss %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))
                    file.write('\n')
    
            print('ep: %03d/%03d | time: %.4f | ML_loss  %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))



def fit_flow_with_delta(nfm, 
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
            y_train_loader=None
            ):
    
    loss_hist = np.array([])

    for ep in range(epochs_flow):

        nfm.train()
        t1 = default_timer()
        loss_flow_epoch = 0
        for _, image in tqdm(enumerate(train_loader)):
            #add noise to the image to use is as conditional image
            if y_train_loader:
                cond_image = image[1].to(device)
                image = image[0]
            else:
                cond_image = add_noise(image).to(device)
            image = image.to(device)
            if image.shape[0] != cond_image.shape[0]:
                cond_image = cond_image[0:image.shape[0]]
            
            delta = batch_delta(nfm=nfm, aem=aeder,
                            X=image, y=cond_image,
                            epsilon=4, alpha=0.1 , num_iter=30,
                            device=device)
                
            y = cond_image
            delta = batch_delta(nfm=nfm, aem=aeder,
                            X=image, y=cond_image,
                            epsilon=4, alpha=0.1 , num_iter=30,
                            device=device).to(device)
            
            print(delta.shape)
            x = aeder.encoder(image, cond_image+delta)
            # Compute loss
            loss_flow = nfm.forward_kld(x, y+delta)
            
            if ~(torch.isnan(loss_flow) | torch.isinf(loss_flow)):
                optimizer_flow.step()
                loss_flow.backward()
            
            # Make layers Lipschitz continuous
            nf.utils.update_lipschitz(nfm, 5)
            
            loss_flow_epoch += loss_flow.item()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss_flow.to('cpu').data.numpy())
            optimizer_flow.zero_grad()
        
        scheduler_flow.step()
        t2 = default_timer()
        loss_flow_epoch /= ntrain
        
        torch.save({
                    'model_state_dict': nfm.state_dict(),
                    'optimizer_state_dict': optimizer_flow.state_dict()
                    }, checkpoint_flow_path)
        
        
        if (ep + 1) % plot_per_num_epoch == 0 or ep + 1 == epochs_flow:
            samples_folder = os.path.join(exp_path, 'Generated_samples')
            if not os.path.exists(samples_folder):
                os.mkdir(samples_folder)
            image_path_generated = os.path.join(
                samples_folder, 'generated')
        
            if not os.path.exists(image_path_generated):
                os.mkdir(image_path_generated)
            sample_number = 9
            ngrid = int(np.sqrt(sample_number))
            
            #####Generating sample data########
            num_samples = torch.tensor(sample_number).to(device)
            y_copy = y[0:sample_number,]
            cond_image_bound = cond_image[0:sample_number]
            y_copy = y_copy.to(device)
            cond_image_bound = cond_image_bound.to(device)
            generated_embed, _ = nfm.sample(y=y_copy, num_samples=num_samples)
            
            generated_samples = aeder.decoder(generated_embed, cond_image_bound).detach().cpu().numpy()
            generated_samples = np.reshape(generated_samples,
                                           [generated_samples.shape[0],
                                            c,image_size, image_size]).transpose(0,2,3,1)

            generated_samples = generated_samples[:sample_number, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)*255.0

     
            cv2.imwrite(os.path.join(image_path_generated, 'epoch %d.png' % (ep,)), generated_samples) 

            
            
            # 

            with open(os.path.join(exp_path, 'results.txt'), 'a') as file:
                    file.write('ep: %03d/%03d | time: %.4f | ML_loss %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))
                    file.write('\n')
    
            print('ep: %03d/%03d | time: %.4f | ML_loss %.4f' %(ep, epochs_flow, t2-t1, loss_flow_epoch))


