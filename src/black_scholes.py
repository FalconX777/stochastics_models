import torch
from brownian import brownian

# Sampling

def bs(s0, mu, sigma, m, n, T, device=None, seed=123):
    """
        Inputs:
            s0: float tensor of shape (m,1), initial prices
            mu: float tensor of shape (m,1), drift rate
            sigma: float tensor of shape (m,1), volatility
            m: int, nb of trajs
            n: int, nb of points
            T: float, trajs generated between [0,T]
            device: torch device
            seed: int
        Outputs:
            dict of torch float tensor of shape (m,n): m B-S trajectories of n points on an uniformly spaced grid on [0,T]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    w = brownian(m, n, T, device, seed)
    t = T/n * torch.ones(1, n, dtype=torch.float, device=device).cumsum(axis=1) - T/n
    s = s0*torch.exp(mu - sigma**2/2)*t + sigma*w)
    rst = {
        'traj':s
    }
    return rst

# Calibration

def bs_param_estim(s, delta_t, n_estim=-1, device=None):
    """
        Inputs:
            s: float tensor of shape (m,n), prices
            delta_t: float, delta time between 2 points of the series
            n_estim: int, nb of indices to use for estimation, -1 to use the whole series
            device: torch device
        Outputs:
            dict of estimated parameters
    """
    if n_estim == -1:
        n_estim = 0
    s0 = s[:,0]
    s = s[:,-n_estim:]

    log_returns = torch.log(s[:,1:]) - torch.log(s[:,:-1])
    sigma = torch.sqrt(log_returns.var(axis=1)/delta_t)
    mu = log_returns.mean(axis=1)/delta_t + sigma**2/2
    params = {
        's0':s0,
        'mu':mu,
        'sigma':sigma
    }
    return params

# Pricing

def bs_call(S, K, T, r, sigma):
    """
        Inputs:
            S: float tensor of shape (m), m spot prices
            K: float tensor of shape (m), m strikes
            T: float tensor of shape (m), m maturities
            r: float tensor of shape (m), m rates
            sigma: float tensor of shape (m), m volatilities
        Outputs:
            torch float tensor of shape (m): m B-S call prices (analytically computed)
    """
    n_cdf = torch.distributions.normal.Normal(0,1).cdf
    d1 = (torch.log(S/K) + r*T)/(sigma*torch.sqrt(T)) + 0.5*sigma*torch.sqrt(T)
    d2 = (torch.log(S/K) + r*T)/(sigma*torch.sqrt(T)) - 0.5*sigma*torch.sqrt(T)
    return S*n_cdf(d1) - torch.exp(-r*T)*K*n_cdf(d2)

def bs_call_MC(S, K, T, r, sigma, n, m_mc, device=None, seed=123):
    """
        Inputs:
            S: float tensor of shape (m), m spot prices
            K: float tensor of shape (m), m strikes
            T: float, maturity
            r: float tensor of shape (m), m rates
            sigma: float tensor of shape (m), m volatilities
            n: int, nb of points per trajectory in [0,T]
            m_mc: int, nb of Monte-Carlo draws
            device: torch device
            seed: int
        Outputs:
            torch float tensor of shape (m): m B-S call prices (Monte-Carlo estimation)
    """
    bs_params = {
        's0':S.unsqueeze(1).repeat(m_mc,1),
        'mu':r.unsqueeze(1).repeat(m_mc,1),
        'sigma':sigma.unsqueeze(1).repeat(m_mc,1),
        'm':S;shape[0]*m_mc,
        'T':T,
        'n':n,
        'device':device,
        'seed':seed
    }
    trajs = bs(**bs_params)['traj']
    Sf = trajs[:,-1].view(m_mc, S.shape[0])
    C_T_K = torch.maximum(Sf-K[:,0].unsqueeze(0), torch.zeros(Sf.shape, device=device)).mean(axis=0)
    return C_T_K

# Implied Volatility

def bs_vega(S, K, T, r, sigma):
    """
        Inputs:
            S: float tensor of shape (m), m spot prices
            K: float tensor of shape (m), m strikes
            T: float tensor of shape (m), m maturities
            r: float tensor of shape (m), m rates
            sigma: float tensor of shape (m), m volatilities
        Outputs:
            torch float tensor of shape (m): m B-S call Vega (analytically computed)
    """
    n_pdf = torch.distributions.normal.Normal(0,1).pdf
    d1 = (torch.log(S/K) + r*T)/(sigma*torch.sqrt(T)) + 0.5*sigma*torch.sqrt(T)
    return S*n_pdf(d1)*torch.sqrt(T)

def bs_IV(target_value, S, K, T, r, max_iter=1000, precision=1e-12, device=None):
    """
        Inputs:
            target_value: float tensor of shape (m), m option prices
            S: float tensor of shape (m), m spot prices
            K: float tensor of shape (m), m strikes
            T: float tensor of shape (m), m maturities
            r: float tensor of shape (m), m rates
            max_iter: int, max number of GD iterations
            precision: float, precision constraint
            device: torch device
        Outputs:
            torch float tensor of shape (m): m implied volatilities
            torch float tensor of shape (m): m (BS call prices with IV)/(target_value)
    """
    
    n_cdf = torch.distributions.normal.Normal(0,1).cdf
    n_pdf = torch.distributions.normal.Normal(0,1).pdf
    eps = 1e-14

    sigma = 0.2*torch.ones(S;shape[0], device=device)
    for i in range(0, max_iter):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        relative_diff = (target_value - price)/(target_value + eps)
        if (torch.max(relative_diff**2) < precision**2):
            return sigma, relative_diff
        new_sigma = sigma + (target_value - price)/(vega+eps)
        sigma = (new_sigma>0)*new_sigma + (new_sigma<=0)*sigma/2
    return sigma, relative_diff