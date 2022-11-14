import torch
import normflow as nf

def real_nvp(latent_dim, K=64, in_res = 64 , c = 3):    
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_dim)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([2*latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        t = nf.nets.MLP([2*latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(
                b, 
                t, 
                s, 
                in_res,
                c,
                latent_dim
                )]
        else:
            flows += [nf.flows.MaskedAffineFlow(
                1 - b, 
                t, 
                s,
                in_res,
                c,
                latent_dim)]
        flows += [nf.flows.ActNorm(latent_dim)]
    
    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(latent_dim)
    
    # Construct flow model
    nfm = nf.CondNormalizingFlow(q0=q0, flows=flows)
    
    return nfm


def glow(latent_dim, K=64, in_res = 64 , c = 3):    
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_dim)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([2*latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        t = nf.nets.MLP([2*latent_dim, 2 * latent_dim, latent_dim], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(
                b, 
                t, 
                s, 
                in_res,
                c,
                latent_dim
                )]
        else:
            flows += [nf.flows.MaskedAffineFlow(
                1 - b, 
                t, 
                s,
                in_res,
                c,
                latent_dim)]
        flows += [nf.flows.ActNorm(latent_dim)]
        if latent_dim > 1:
            flows += [nf.flows.Invertible1x1Conv(latent_dim)]
    
    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(latent_dim)
    
    # Construct flow model
    nfm = nf.CondNormalizingFlow(q0=q0, flows=flows)
    
    return nfm