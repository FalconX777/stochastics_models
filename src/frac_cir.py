import torch
import numpy as np
from frac_brownian import frac_brownian

# Sampling 

def fcir(r0, a, b, sigma, H, m, n, T, device=None, seed=123):
    """
        Inputs:
            r0: float tensor of shape (m,1), initial rates
            a: float tensor of shape (m), long term mean level
            b: float tensor of shape (m), speed of reversion
            sigma: float tensor of shape (m), instantaneous volatility
            H: float, Hurst exponent
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            dict of torch float tensor of shape (m,n): m fCIR trajectories of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    fBm = frac_brownian(H, m, n, T, device, seed)
    fBm_increments = fBm[:,1:] - fBm[:,:-1]

    r = r0*torch.ones(m, n, dtype=torch.float, device=device)
    for i in range(1,n):
        r[:,i] = r[:,i-1] + a*delta_t*(b-r[:,i-1]) + sigma*torch.sqrt(torch.abs(r[:,i-1]))*fBm_increments[:,i-1]
    res = {
        'traj':r
    }
    return res