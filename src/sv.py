import torch
import numpy as np

# Sampling 

def sv(s0, x0, mu, nu, alpha, m_sigma, rho, m, n, T, device=None, seed=123):
    """
        Inputs:
            s0: float tensor of shape (m,1), initial prices
            x0: float tensor of shape (m,1), initial vol
            mu: float tensor of shape (m), drift rate
            nu: float tensor of shape (m), vol of vol
            alpha: float tensor of shape (m), reversion frequency
            m_sigma: float tensor of shape (m), mean log-volatility
            rho: float tensor of shape (m), correlation of the brownian motions
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            dict of 
                torch float tensor of shape (m,n): m SV trajectories of the price s, of n points on an uniformly spaced grid on [0,T]
                torch float tensor of shape (m,n): m SV trajectories of the latent process X, of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian_s = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()
    gaussian_X = rho*gaussian_s + np.sqrt(1-rho**2)*sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()

    s = s0*torch.ones(m, n, dtype=torch.float, device=device)
    x = x0*torch.ones(m, n, dtype=torch.float, device=device)
    for i in range(1,n):
        x[:,i] = x[:,i-1] + nu*gaussian_x[:,i-1] - alpha*delta_t*(x[:,i-1]-m_sigma)
        s[:,i] = s[:,i-1] + mu*s[:,i-1]*delta_t + torch.exp(x[:,i])*s[:,i-1]*gaussian_s[:,i-1]
    res = {
        'traj':s,
        'vol':x,
    }
    return res

# Calibration

def sv_param_estim(s, V, r, delta_t, n_estim=-1, device=None):
    """
        Inputs:
            s: float tensor of shape (m,n), prices
            V: float tensor of shape (m,n), squared volatilities (V_t = exp(2*X_t))
            r: float tensor of shape (m), average yield of the equity
            delta_t: float, delta time between 2 points of the series
            n_estim: int, nb of indices to use for estimation, -1 to use the whole series
            device: torch device
        Outputs:
            dict of estimated parameters using MLE
    """
    if n_estim == -1:
        n_estim = 0
    s = s[:,-n_estim:]
    V = V[:,-n_estim:]
    s0 = s[:,0]
    x0 = torch.log(V[:,0])/2
    m = s.shape[1]
    n = s.shape[1]

    X = torch.log(V)/2
    mean_X = X[:,:-1].mean(axis=1)
    mean_sqV = torch.sqrt(V[:,1:]/V[:,:-1]).mean(axis=1)

    # Parameter estimation
    alpha = (torch.sqrt(V[:,1:]/V[:,:-1])*X[:,:-1]).mean(axis=1) - mean_sqV*mean_X
    alpha /= (mean_X**2 - (X[:,:-1]**2).mean(axis=1))*delta_t
    eq2_param = mean_sqV + alpha*mean_X*delta_t
    nu = torch.sqrt(((V[:,1:]/V[:,:-1]).mean(axis=1) - 2*eq2_param*mean_sqV + 2*alpha*delta_t*(torch.sqrt(V[:,1:]/V[:,:-1])*X[:,:-1]).mean(axis=1) + ((eq2_param - alpha*delta_t*X[:,:-1])**2).mean(axis=1))/delta_t)
    m_sigma = (eq2_param - 1 - nu**2*delta_t/2)/alpha/delta_t
    Delta_W1 = torch.sqrt(V[:,:-1])*(torch.log(s[:,1:]/s[:,:-1]) - (r.unsqueeze(1) - V[:,:-1]/2)*delta_t)
    Delta_W2 = (X[:,1:] - X[:,:-1] + alpha*(X[:,:-1] - m_sigma)*delta_t)/nu
    rho = (Delta_W1*Delta_W2).mean(axis=1)/delta_t

    params = {
        's0':s0,
        'x0':x0,
        'mu':r,
        'nu':nu,
        'alpha':alpha,
        'm_sigma':m_sigma,
        'rho':rho
    }
    return params