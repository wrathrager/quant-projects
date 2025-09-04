# implied volatility solvers + helpers
# src/implied_vol.py
import numpy as np
from scipy.optimize import brentq
from src.bs import bs_price

def implied_vol_from_price(mkt_price, S, K, r, q, T, option='call', tol=1e-8):
    # define objective: model_price(sigma) - mkt_price
    def objective(sigma):
        return bs_price(S,K,r,q,sigma,T,option) - mkt_price

    # bounds
    low, high = 1e-8, 5.0
    # Check if market price is within [price(sigma_low), price(sigma_high)]
    try:
        # bracketing
        return brentq(objective, low, high, xtol=tol, maxiter=200)
    except Exception as e:
        return np.nan  # handle by logging in notebook
