import torch
from frac_brownian import frac_brownian
import numpy as np

# Sampling 

def rfsv(s0, x0, mu, nu, alpha, m_sigma, H, m, n, T, device=None, seed=123):
    """
        Inputs:
            s0: float tensor of shape (m,1), initial prices
            x0: float tensor of shape (m,1), initial vol
            mu: float tensor of shape (m), drift rate
            nu: float tensor of shape (m), vol of vol
            alpha: float tensor of shape (m), reversion frequency
            m_sigma: float tensor of shape (m), mean log-volatility
            H: float, Hurst exponent
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs: (https://arxiv.org/pdf/1410.3394.pdf)
            dict of 
                torch float tensor of shape (m,n): m rfSV trajectories of the price s, of n points on an uniformly spaced grid on [0,T]
                torch float tensor of shape (m,n): m rfSV trajectories of the latent process X, of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()
    fBm = frac_brownian(H, m, n, T, device, seed=None)
    fBm_increments = fBm[:,1:] - fBm[:,:-1]

    s = s0*torch.ones(m, n, dtype=torch.float, device=device)
    x = x0*torch.ones(m, n, dtype=torch.float, device=device)
    for i in range(1,n):
        x[:,i] = x[:,i-1] + nu*fBm_increments[:,i-1] - alpha*delta_t*(x[:,i-1]-m_sigma)
        s[:,i] = s[:,i-1] + mu*s[:,i-1]*delta_t + torch.exp(x[:,i])*s[:,i-1]*gaussian[:,i-1]
    res = {
        'traj':s,
        'vol':x,
    }
    return res