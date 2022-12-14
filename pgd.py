import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os 

def norms(Z):
    """Return the norms of the input tensor over all but the first dimension """
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None] 
def norms_latent_space(Z):
    """Return the norms of the input tensor over all but the first dimension """
    return Z.view(Z.shape[0], -1).norm(dim=0) 

def batch_delta(
    nfm, aem, X, y, epsilon, alpha, num_iter,device):
    """Construct PGD adversial examples on the examples X"""
    X = X.to(device)
    y = y.to(device)
    assert X.shape == y.shape, f"X and y should have the same shape, but they are {X.shape} and {y.shape}"
    delta = torch.zeros_like(X, requires_grad=True)
    for i in range(num_iter):
        z_delta, log_q_delta = nfm.sample(y=y+delta, num_samples=y.shape[0])
        z, log_q = nfm.sample(y=y, num_samples=y.shape[0])
        x_hat_delta = aem.decoder(z_delta, y+delta)
        x_hat = aem.decoder(z, y)
        loss = F.mse_loss(x_hat_delta, x_hat)
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data  = torch.min(torch.max(delta.detach(), -(y+delta)), 1-(y+delta))
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()     
    return delta.detach()


# def fgsm(nfm, aem, X, y, mean_sample, epsilon, alpha, num_iter,device):
#     """Construct FGSM adversial examples on the examples X"""
#     X = X.to(device)
#     y = y.to(device)
#     assert X.shape == y.shape, f"X and y should have the same shape, but they are {X.shape} and {y.shape}"
#     assert X.shape[0] == 1, "Only one sample is allowed"
#     delta = torch.zeros_like(X, requires_grad=True)
#     if mean_sample:
#         z_delta, log_q_delta = nfm.sample_mu(y=y+delta)
#         z, log_q = nfm.sample_mu(y=y)        
#     else:
#         z_delta, log_q_delta = nfm.sample(y=y+delta)
#         z, log_q = nfm.sample(y=y)
#     x_hat_delta = aem.decoder(z_delta, y+delta)
#     x_hat = aem.decoder(z, y+delta)
#     loss = F.mse_loss(x_hat_delta, x_hat)
#     loss.backward()
#     delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
#     delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
#     delta.grad.zero_()
#     return delta.detach()

def fgsm(nfm, aem, X, y, mean_sample,epsilon, alpha, num_iter,device, exp_path):
    """Construct FGSM adversial examples on the examples X"""
    loss_hist = np.array([])
    X = X.to(device)
    y = y.to(device)
    assert X.shape == y.shape, f"X and y should have the same shape, but they are {X.shape} and {y.shape}"
    assert X.shape[0] == 1, "Only one sample is allowed"
    delta = torch.zeros_like(X, requires_grad=True)
    prev_delta = None
    for i in range(num_iter):
        # if i >5:
        #     alpha = 0.1
        #     # epsilon = 4
        # elif i > 100:
        #     alpha = 0.1
        if mean_sample:
            z_delta, log_q_delta = nfm.sample_mu(y=y+delta)
            z, log_q = nfm.sample_mu(y=y)        
        else:
            z_delta, log_q_delta = nfm.sample(y=y+delta)
            z, log_q = nfm.sample(y=y)
        x_hat_delta = aem.decoder(z_delta, y+delta)
        x_hat = aem.decoder(z, y)
        loss = F.mse_loss(x_hat_delta, x_hat)
        loss.backward()
        # print("Train the delta with L2 norm", loss)
        # # delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        # delta.data += alpha*delta.grad #/ norms(delta.grad.detach())
        # # delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        # # delta.data *= epsilon
        # delta.grad.zero_()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data  = torch.min(torch.max(delta.detach(), -(y+delta)), 1-(y+delta))
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        loss_hist = np.append(loss_hist, loss.detach().cpu().numpy())
        print('ep: %03d/%03d | delta_loss %.4f ' %(i, num_iter, loss))
    plt.plot(np.arange(num_iter), loss_hist, )
    plt.title('Delta loss')
    plt.xlabel('iterations')
    plt.ylabel('MSE loss')
    if mean_sample:
        plt.savefig(os.path.join(exp_path, 'delta_loss_mu.jpg'))
        np.save(os.path.join(exp_path, 'delta_loss_mu.npy'), loss_hist)
    else:
        plt.savefig(os.path.join(exp_path, 'delta_loss.jpg'))
        np.save(os.path.join(exp_path, 'delta_loss.npy'), loss_hist)
    plt.close()
        
    return delta.detach()
def fgsm_relnvp(nfm, aem, X, y, mean_sample,epsilon, alpha, num_iter,device, exp_path):
    """Construct FGSM adversial examples on the examples X"""
    loss_hist = np.array([])
    X = X.to(device)
    y = y.to(device)
    assert X.shape == y.shape, f"X and y should have the same shape, but they are {X.shape} and {y.shape}"
    assert X.shape[0] == 1, "Only one sample is allowed"
    delta = torch.zeros_like(X, requires_grad=True)
    first = True
    for i in range(num_iter):
        if mean_sample:
            z_delta, log_q_delta = nfm.sample_mu(y=y+delta)
            z, log_q = nfm.sample_mu(y=y)        
        else:
            z_delta, log_q_delta = nfm.sample(y=y+delta)
            z, log_q = nfm.sample(y=y)
        loss = F.cross_entropy(z_delta,z)
        # loss = nfm.forward_kld(z_delta, y)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            delta.grad = torch.nan_to_num(delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(delta.grad).any() or torch.isinf(delta.grad).any():
                print("nan in delta grad")
            # norms_val = norms(delta.grad.detach())
            # norms_val = norms_val.clamp(min=1e-9)
            print(norms(delta.grad.detach()).clamp(min=0e-2))
            if first:
                first = False
                grad_norm = alpha*delta.grad.detach() / norms(delta.grad.detach()).clamp(min=1e-5)
            grad_norm = alpha*delta.grad.detach() / norms(delta.grad.detach())
            if torch.isnan(grad_norm).any() :
                # grad_norm = torch.nan_to_num(grad_norm, nan=0e9, posinf=0.1, neginf=-0.1)
                print("nan in delta before")
                print(grad_norm)
            
            delta.data = torch.min(torch.max(delta.detach(), -y), 1-y)
            delta.data += grad_norm
            # delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            # if torch.isnan(delta).any() :
            #     print("nan in delta")
            #     print(delta)
            # i                                                                                                                # delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
        print('ep: %03d/%03d | delta_loss %.4f ' %(i, num_iter, loss))
        loss_hist = np.append(loss_hist, loss.detach().cpu().numpy())
    plt.plot(np.arange(num_iter), loss_hist, )
    plt.title('Delta loss')
    plt.xlabel('iterations')
    plt.ylabel('MSE loss')
    if mean_sample:
        plt.savefig(os.path.join(exp_path, 'delta_loss_mu.jpg'))
        np.save(os.path.join(exp_path, 'delta_loss_mu.npy'), loss_hist)
    else:
        plt.savefig(os.path.join(exp_path, 'delta_loss.jpg'))
        np.save(os.path.join(exp_path, 'delta_loss.npy'), loss_hist)
    plt.close()
            
    return delta.detach()

    
def pgd_l2(nf_model,ae_model, Y, epsilon, alpha, num_iter, device):
    """Construct FGSM adversial examples on the examples X"""
    Y_truth = Y[0].to(device)
    Y_cond = Y[1].to(device)
    assert Y_truth.shape == Y_cond.shape, f"Y_truth and Y_cond should have the same shape, but they are {Y_truth.shape} and {Y_cond.shape}"
    assert Y_truth.shape[0] == 1, "Only one sample is allowed"
    delta = torch.zeros_like(Y_truth, requires_grad=True)
    print("Train the delta with L2 norm")
    for t in range(num_iter):
        #Model that is generating the sample and the loss to calculate pertubation
        z, log_q = nf_model.sample(y=Y_cond+delta)
        x_hat = ae_model.decoder(z, Y_cond+delta)
        assert Y_truth.shape == x_hat.shape, f"Y_truth and y_hat should have the same shape, but they are {Y_truth.shape} and {y_hat.shape}" 
        loss = -F.mse_loss(x_hat, Y_truth)
        if ~(torch.isnan(loss).any() | torch.isinf(loss).any()):
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data  = torch.min(torch.max(delta.detach(), -Y_truth), 1-Y_truth)
            # delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
        if t % 10 == 0:
            print('ep: %03d/%03d | delta_loss %.4f ' %(t, num_iter, loss ))        
        
    return delta.detach()

def pgd_l2_dataset(nf_model,ae_model, Ytr, epsilon, alpha, num_iter, device):
    """Construct FGSM adversial examples on the examples X"""
    Y = next(iter(Ytr))
    Y_truth = Y[0].to(device)
    Y_cond = Y[1].to(device)
    assert Y_truth.shape == Y_cond.shape, f"Y_truth and Y_cond should have the same shape, but they are {Y_truth.shape} and {Y_cond.shape}"
    delta = torch.zeros_like(Y_truth, requires_grad=True)
    print("Train the delta with L2 norm")
    for t,Y in enumerate(Ytr):
        #Model that is generating the sample and the loss to calculate pertubation
        Y_truth = Y[0].to(device)
        Y_cond = Y[1].to(device)
        for _ in range(10):
            z, log_q = nf_model.sample(
                y=Y_cond+delta, 
                num_samples=torch.tensor(Y_truth.shape[0]).to(device)
                )
            y_hat = ae_model.decoder(z, Y_cond)
            assert Y_truth.shape == y_hat.shape, f"Y_truth and y_hat should have the same shape, but they are {Y_truth.shape} and {y_hat.shape}" 
            loss = F.mse_loss(y_hat, Y_truth)
            if ~(torch.isnan(loss).any() | torch.isinf(loss).any()):
                loss.backward()
                delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
                delta.data  = torch.min(torch.max(delta.detach(), -Y_truth), 1-Y_truth)
                delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
                delta.grad.zero_()
        if t % 10 == 0:
            print('ep: %03d/%03d | delta_loss %.4f ' %(t, (len(Ytr)), loss ))        
    
    return delta.detach()


def pgd_l2_z_noise(nf_model,ae_model, Y, latent_space, epsilon, alpha, num_iter, device):
    """This function is used to generate the noise to add to the latent space"""
    Y_truth = Y[0].to(device)
    Y_cond = Y[1].to(device)
    assert Y_truth.shape == Y_cond.shape, f"Y_truth and Y_cond should have the same shape, but they are {Y_truth.shape} and {Y_cond.shape}"
    assert Y_truth.shape[0] == 1, "Only one sample is allowed"
    delta  = torch.zeros_like(Y_truth, requires_grad=True)
    print("Train the delta with L2 norm")
    for t in range(num_iter):
        z, log_q = nf_model.sample(y=Y_cond)
        z_delta, _ = nf_model.sample(y=Y_cond+delta)
        assert z.shape == z_delta.shape, f"z and z_delta should have the same shape, but they are {z.shape} and {z_delta.shape}"
        loss = F.mse_loss(z_delta, z)
        if ~(torch.isnan(loss).any() | torch.isinf(loss).any()):
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data  = torch.min(torch.max(delta.detach(), -z), 1-z)
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
        if t % 10 == 0:
            print('ep: %03d/%03d | delta_loss %.4f ' %(t, num_iter, loss ))
    return delta.detach()

def pgd(nf_model,ae_model,loss, Y, epsilon, alpha, num_iter, device):
    """Construct FGSM adversial examples on the examples X"""
    Y_truth = Y[0].to(device)
    Y_cond = Y[1].to(device)
    assert Y_truth.shape == Y_cond.shape, f"Y_truth and Y_cond should have the same shape, but they are {Y_truth.shape} and {Y_cond.shape}"
    assert Y_truth.shape[0] == 1, "Only one sample is allowed"
    delta = torch.zeros_like(Y_truth, requires_grad=True)
    for t in range(num_iter):
        #TODO I should use MSE 
        z, log_q = nf_model.sample(y=Y_cond+delta)
        y_hat = ae_model.decoder(z, Y_cond)
        loss = F.mse_loss(y_hat, Y_truth)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta

