# implied volatility solvers + helpers
# src/implied_vol.py
from scipy.stats import norm
import numpy as np
from bs import black_scholes_price, vega

def implied_volatility(S, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=100):
    if T <= 0 or S <= 0 or K <= 0 or market_price < 0:
        return None
    sigma = 0.2  # initial guess
    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v == 0 or np.isnan(v) or np.isinf(v):
            break
        sigma += diff / v
        # Clamp sigma to reasonable bounds to avoid overflow/underflow
        if sigma < 1e-6:
            sigma = 1e-6
        if sigma > 5.0:
            sigma = 5.0
        if np.isnan(sigma) or np.isinf(sigma):
            return None
    return None  # did not converge

