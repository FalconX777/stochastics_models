import torch
import numpy as np

# Sampling 

def cir(r0, a, b, sigma, m, n, T, device=None, seed=123):
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
            dict of torch float tensor of shape (m,n): m CIR trajectories of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()

    r = r0*torch.ones(m, n, dtype=torch.float, device=device)
    for i in range(1,n):
        r[:,i] = r[:,i-1] + a*delta_t*(b-r[:,i-1]) + sigma*torch.sqrt(torch.abs(r[:,i-1]))*gaussian[:,i-1]
    res = {
        'traj':r
    }
    return res

# Calibration

def cir_param_estim(r, delta_t, n_estim=-1, device=None):
    """
        Inputs:
            r: float tensor of shape (m,n), rates
            delta_t: float, delta time between 2 points of the series
            n_estim: int, nb of indices to use for estimation, -1 to use the whole series
            device: torch device
        Outputs:
            dict of estimated parameters using ordinary least squares (http://lnu.diva-portal.org/smash/get/diva2:1270329/FULLTEST01.pdf)
    """
    if n_estim == -1:
        n_estim = 0
    r0 = r[:,0]
    r = r[:,-n_estim:]
    n = r.shape[1]
    eps = 1e-14

    Y = (r[:,1:] - r[:,:-1])/torch.sqrt(torch.abs(r[:,:-1])+eps)
    Y = Y.unsqueeze(2)
    Z = torch.empty(Y.shape[0], Y.shape[1], 2, dtype=torch.float, device=device)
    Z[:,:,0] = delta_t/torch.sqrt(torch.abs(r[:,:-1])+eps)
    Z[:,:,1] = delta_t*torch.sqrt(torch.abs(r[:,:-1])+eps)
    Z_pseudo_inv = torch.linalg.pinv(Z)
    beta = Z_peudo_inv @ Y

    a = -beta[:,1,0]
    b = beta[:,0,0]/a 
    sigma = (torch.sqrt(((Y - Z @ beta)**2).sum(axis=1))/np.sqrt(n*delta_t))[:,0]
    params = {
        'r0':r[:,0],
        'a':a,
        'b':b,
        'sigma':sigma
    }
    return params