import torch

# Sampling

def integ1(v, i, delta_t, H, lam, theta, device):
    integ = torch.zeros(v.shape[0], dtype=torch.float, device=device)
    for j in range(i):
        integ += (delta_t*(i-j))**(H-1/2)*lam*(theta-v[:,j])*delta_t
    return integ

def integ2(v, i, delta_t, H, gaussian_b, device):
    integ = torch.zeros(v.shape[0], dtype=torch.float, device=device)
    for j in range(i):
        integ += (delta_t*(i-j))**(H-1/2)*torch.sqrt(v[:,j])*gaussian_b[:,j]
    return integ

def rheston(s0, v0, mu, theta, lam, nu, H, rho, m, n, T, device=None, seed=123):
    """
        Inputs:
            s0: float tensor of shape (m,1), initial prices
            v0: float tensor of shape (m,1), initial vol
            mu: float tensor of shape (m), long term mean level
            theta: float tensor of shape (m), speed of reversion
            lam: float tensor of shape (m)
            nu: float tensor of shape (m)
            H: float, Hurst exponent
            rho: float tensor of shape (m,1), correlation of the brownian motions
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            dict of 
                torch float tensor of shape (m,n): m rough Heston trajectories of the price s, of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian_w = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()
    gaussian_b = rho*gaussian_w + np.sqrt(1-rho**2)*sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()
    
    log_s = np.log(s0)*torch.ones(m, n, dtype=torch.float, device=device)
    v = v0*torch.ones(m, n, dtype=torch.float, device=device)
    for i in range(1,n):
        v[:,i] += integ1(v, i, delta_t, H, lam, theta, device) + lam*nu*integ2(v, i, delta_t, H, gaussian_b, device)
        log_s[:,i] = log_s[:,i-1] + mu*delta_t + torch.sqrt(v[:,i])*gaussian_w[:,i-1]
    rst = {
        'traj':torch.exp(log_s)
    }
    return rst