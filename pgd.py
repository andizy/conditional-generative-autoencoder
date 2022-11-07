import torch
import torch.nn.functional as F

def norms(Z):
    """Return the norms of the input tensor over all but the first dimension """
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None] 


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
            print('ep: %03d/%03d | delta_loss %.4f ' %(t, num_iter, loss ))        
        
    return delta.detach()


#TODO Fix at single sample and calculate the adversial pertubation noise
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

