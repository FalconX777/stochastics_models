import torch

def brownian(m, n, T, device=None, seed=123):
    """
        Inputs:
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            torch float tensor of shape (m,n): m brownian motion trajectories of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    delta_t = T/n*torch.ones(1, dtype=torch.float, device=device)
    sq_dt = torch.sqrt(delta_t)
    gaussian = sq_dt*torch.empty(m, n-1, dtype=torch.float, device=device).normal_()
    w = gaussian.cumsum(axis=1)
    res = torch.zeros(m, n, dtype=torch.float, device=device)
    res[:,1:] = w
    return res