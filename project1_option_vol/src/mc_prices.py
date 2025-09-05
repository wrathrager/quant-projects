# Monte Carlo pricers: Asian, Barrier
# src/mc_pricers.py
import numpy as np
from math import sqrt
from .gbm import gbm_paths  # expects function that returns paths or implement inline

def _generate_gbm_paths(S0, r, sigma, T, M, N, seed=None, antithetic=False):
    """
    Generate GBM paths using log-Euler exact discretization.
    Returns array shape (N, M+1) including S0 at index 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / M
    nudt = (r - 0.5 * sigma**2) * dt
    sigsdt = sigma * sqrt(dt)

    if antithetic:
        half = N // 2
        Z = rng.standard_normal((M, half))
        Z = np.concatenate([Z, -Z], axis=1)
    else:
        Z = rng.standard_normal((M, N))

    # cumulative sums of log increments
    log_increments = nudt + sigsdt * Z
    log_paths = np.cumsum(log_increments, axis=0)  # shape (M, N)
    S_paths = np.exp(log_paths)
    S_paths = np.vstack([np.ones((1, N)), S_paths])  # prepend ones for S0 multiplier
    S_paths = (S0 * S_paths).T  # shape (N, M+1)
    return S_paths

def mc_arithmetic_asian(S0, K, r, sigma, T, M=100, N=20000, option="call",
                        antithetic=False, control_variate=False, seed=None):
    """
    Monte Carlo price for arithmetic-average Asian option.
    control_variate: uses European payoff on ST as control (BS price must be known externally;
    here we use sample to form control with known analytic expectation replaced by BS price if provided).
    For robust usage, supply european_price parameter externally and adjust.
    """
    # generate paths
    S_paths = _generate_gbm_paths(S0, r, sigma, T, M, N, seed=seed, antithetic=antithetic)
    # arithmetic average excluding S0: average of M future points
    S_avg = S_paths[:, 1:].mean(axis=1)

    if option == "call":
        payoff = np.maximum(S_avg - K, 0.0)
    else:
        payoff = np.maximum(K - S_avg, 0.0)

    # control variate using European payoff on ST (final price)
    if control_variate:
        ST = S_paths[:, -1]
        european_payoff = np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)
        # compute sample covariance and optimal coef b = Cov(payoff, control)/Var(control)
        cov = np.cov(payoff, european_payoff, ddof=0)
        cov_xy = cov[0,1]
        var_y = cov[1,1]
        if var_y > 0:
            b_hat = cov_xy / var_y
            # need expected value of control under risk-neutral: E[european_payoff] = BS_price(ST-based)
            # We don't have closed-form BS price here for all params, so we use sample mean of european_payoff discounted
            # A better approach: pass analytic BS price from caller and use that as E[Y]. Here we use sample mean as fallback.
            EY = np.mean(european_payoff)
            adj_payoff = payoff - b_hat * (european_payoff - EY)
            estimate = np.exp(-r*T) * np.mean(adj_payoff)
            # Also compute standard error
            stderr = np.exp(-r*T) * np.std(adj_payoff, ddof=1) / np.sqrt(len(adj_payoff))
            return estimate, stderr
    # no control variate
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price, stderr

def mc_barrier(S0, K, r, sigma, T, B, M=100, N=20000, option="call",
               barrier_type="up-and-out", antithetic=False, seed=None):
    """
    Monte Carlo pricing for discrete-monitoring barrier options.
    barrier_type: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
    """
    S_paths = _generate_gbm_paths(S0, r, sigma, T, M, N, seed=seed, antithetic=antithetic)
    if option == "call":
        payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0.0)

    if "up" in barrier_type:
        breached = (S_paths[:, 1:].max(axis=1) >= B)
    else:
        breached = (S_paths[:, 1:].min(axis=1) <= B)

    if "out" in barrier_type:
        payoff[breached] = 0.0
    else:  # "in"
        payoff[~breached] = 0.0

    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(len(payoff))
    return price, stderr
