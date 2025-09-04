# implied volatility solvers + helpers
# src/implied_vol.py
from scipy.stats import norm
import numpy as np
from bs import black_scholes_price, vega

def implied_volatility(S, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=100):
    sigma = 0.2  # initial guess
    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = market_price - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v == 0:
            break
        sigma += diff / v
    return None  # did not converge

