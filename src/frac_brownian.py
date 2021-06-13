import torch
import numpy as np

# Sampling

def frac_brownian(H, m, n, T, device=None, seed=123):
    """
        Inputs:
            H: float, Hurst exponent
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            torch float tensor of shape (m,n): m fractional brownian motion trajectories of n points on an uniformly spaced grid on [0,T], using FFT (https://arxiv.org/pdf/1303.1648.pdf)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    Lp = 2*n
    xi = torch.empty(m, Lp, dtype=torch.float, device=device).normal_()

    delta_t = T/n 
    C = np.array([abs(delta_t*(m+1))**(2*H)/2 - abs(delta_t*m)**(2*H) + abs(delta_t*(m-1))**(2*H)/2 for m in range(Lp//2+1)])
    C = torch.from_numpy(C).to(device)
    C_rm = torch.empty(Lp, dtype=torch.float, device=device)
    C_rm[:Lp//2] = C[:Lp//2]
    idx = [i for i in range(Lp//2, 0, -1)]
    C_rm[Lp//2:] = C[idx]

    hat_c = torch.fft.fft(C_rm, n=None, dim=-1, norm='backward')
    hat_delta = torch.sqrt(Lp*hat_c)*xi
    delta = torch.fft.ifft(hat_delta, n=None, dim=-1, norm='backward')
    delta_x = delta.real + delta.imag

    x = torch.zeros(m, n, dtype=torch.float, device=device)
    x[:,1:] = delta_x.cumsum(axis=1)[:,:n-1]

    return x

# Calibration - IR estimator for Hurst exponent H

def rho(H):
    return (-3**(2*H) + 2**(2*H+2) - 7)/(8 - 2**(2*H+1))

def lam(x):
    return torch.accos(-x)/np.pi + torch.sqrt((1+x)/(1-x))*torch.log(2/(1+x))/np.pi

def grad_rho(H):
    part1 = (-2*np.log(3)*3**(2*H) + 2*np.log(2)*2**(2*H+2))/(8 - 2**(2*H+1))
    part2 = 2*np.log(2)*2**(2*H+1)*(-3**(2*H+2) - 7)/(8 - 2**(2*H+1))**2
    return part1 + part2

def grad_lam(x):
    return 1/np.pi/torch.sqrt(1-x**2) + 1/np.pi*(torch.log(2/(1+x))*(1-x)/(torch.sqrt(1-x**2)) - 1/torch.sqrt(1-x**2))

def l_rho(H):
    return lam(rho(H))

def grad_l_rho(H):
    return grad_rho(H)*grad_lam(rho(H))

def fBm_IR_param_estim(s, delta_t, approx_order=2, n_estim=-1, device=None):
    """
        Inputs:
            s: float tensor of shape (m,n), m trajectories of n points
            delta_t: float, delta time between 2 points of the series
            approx_order: int, approx order of lambda(rho(H)), 1 for linear, 2 for quadratic, every other value for Newton method with quadratic approx initialization
            n_estim: int, nb of indices to use for estimation, -1 to use the whole series
            device: torch device
        Outputs:
            dict of estimated parameters with IR estimator (http://www.ressources-actuarielles.net/EXT/ISFA/1226-02.nsf/0/174467a2b647a141c1257a1d006cf9c6/$FILE/memoire_IA.pdf)
    """
    if n_estim == -1:
        n_estim = 0
    s = s[:,-n_estim:]
    m = s.shape[0]
    n = s.shape[1]
    eps = 1e-14

    Delta1X = s[:,1:] - s[:,:-1]
    Delta2X = Delta1X[:,1:] - Delta1X[:,:-1]
    R2 = (torch.abs(Delta2X[:,:-1] + Delta2X[:,1:] + eps)/(torch.abs(Delta2X[:,:-1]) + torch.abs(Delta2X[:,1:]) + eps)).mean(axis=1)
    if approx_order == 1:
        # Linear Approx
        H = (R2 - 0.5174)/0.1468
    elif approx_order == 2:
        # Quadratic approx
        delta = torch.sqrt(0.115**2 - 4*0.0312*(0.5228 - R2))
        H = (-0.115 + delta)/2/0.0312
    else:
        # Newton method with Quadratic approx ini
        delta = torch.sqrt(0.115**2 - 4*0.0312*(0.5228 - R2))
        H = (-0.115 + delta)/2/0.0312
        prev_H = H - 1
        n_iter = 0
        while n_iter < 1000 and torch.max(torch.abs(prev_H-H)) > 1e-8:
            prev_H = HurstH -= (l_rho(H)-R2)/grad_l_rho(H)
    fBm_params = {
        'H':H 
    }
    return fBm_params