import torch
import numpy as np

# Sampling 

def vasicek(r0, a, b, sigma, m, n, T, device=None, seed=123):
    """
        Inputs:
            r0: float tensor of shape (m,1), initial rates
            a: float tensor of shape (m), long term mean level
            b: float tensor of shape (m), speed of reversion
            sigma: float tensor of shape (m), instantaneous volatility
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            dict of torch float tensor of shape (m,n): m Vasicek trajectories of n points on an uniformly spaced grid on [0,T], computed with continuous formula (integral approximated with Euler formula, further orders do not giving significantly better results)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()

    t = T/n*torch.ones(m, n, dtype=torch.float, device=device).cumsum(axis=1) - T/n 
    exp_at = torch.exp(a*t)
    exp_mat = torch.exp(-a*t)
    integral = ((exp_at[:,1:]-exp_at[:,:-1])/a*gaussian).cumsum(axis=1)/delta_t
    r = r0*exp_mat + b*(1-exp_mat)
    r[:,1:] += sigma*exp_mat[:,1:]*integral
    res = {
        'traj':r
    }
    return res

# Calibration

def vasicek_param_estim(r, delta_t, n_estim=-1, device=None):
    """
        Inputs:
            r: float tensor of shape (m,n), rates
            delta_t: float, delta time between 2 points of the series
            n_estim: int, nb of indices to use for estimation, -1 to use the whole series
            device: torch device
        Outputs:
            dict of estimated parameters using MLE (https://odr.chalmers.se/bitstream/20.500.12380/256885.pdf)
    """
    if n_estim == -1:
        n_estim = 0
    r0 = r[:,0]
    r = r[:,-n_estim:]
    n = r.shape[1]

    a = (-1/delta_t)*(torch.log(n*(r[:,1:]*r[:,:-1]).sum(axis=1) - r[:,1:].sum(axis=1)*r[:,:-1].sum(axis=1)) - torch.log(n*(r[:,:-1]**2).sum(axis=1) - r[:,:-1].sum(axis=1)**2))
    a = (a - 4/delta_t/n)*n/(n+2)
    b = 1/(n*(1-torch.exp(-a*delta_t)))*(r[:,1:].sum(axis=1) - torch.exp(a*delta_t)*r[:,:-1].sum(axis=1))
    sigma = 2*a/(n*(1-torch.exp(-2*a*delta_t)))*(r[:,1:] - torch.exp(-a*delta_t).unsqueeze(1)*r[:,:-1] - (b*(1-torch.exp(-a*delta_t))).unsqueeze(1)).sum(axis=1)**2
    params = {
        'r0':r[:,0],
        'a':a,
        'b':b,
        'sigma':sigma
    }
    return params