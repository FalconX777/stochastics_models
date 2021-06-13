import torch
import numpy as np
from frac_brownian import frac_brownian

# Sampling 

def fOU(r0, a, b, sigma, H, m, n, T, device=None, seed=123):
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
            dict of torch float tensor of shape (m,n): m fractional O-U trajectories of n points on an uniformly spaced grid on [0,T], computed with continuous formula (integral approximated with Euler formula, further orders do not giving significantly better results)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    fBm = frac_brownian(H, m, n, T, device, seed)
    fBm_increments = fBm[:,1:] - fBm[:,:-1]

    t = T/n*torch.ones(m, n, dtype=torch.float, device=device).cumsum(axis=1) - T/n 
    exp_at = torch.exp(a*t)
    exp_mat = torch.exp(-a*t)
    integral = ((exp_at[:,1:]-exp_at[:,:-1])/a*fBm_increments).cumsum(axis=1)/delta_t
    r = r0*exp_mat + b*(1-exp_mat)
    r[:,1:] += sigma*exp_mat[:,1:]*integral
    res = {
        'traj':r
    }
    return res