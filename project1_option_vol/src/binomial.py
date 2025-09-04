# CRR binomial tree implementation
# src/binomial.py
import numpy as np

def crr_price(S, K, r, q, sigma, T, N=200, option='call', american=False):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    disc = np.exp(-r*dt)

    # terminal prices
    j = np.arange(N+1)
    ST = S * (u**j) * (d**(N-j))
    if option=='call':
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # backward induction
    for i in range(N-1, -1, -1):
        values = disc * (p*values[1:] + (1-p)*values[:-1])
        if american:
            # compute exercise values
            j = np.arange(i+1)
            ST_i = S * (u**j) * (d**(i-j))
            exercise = np.maximum(ST_i - K, 0.0) if option=='call' else np.maximum(K - ST_i, 0.0)
            values = np.maximum(values, exercise)
    return values[0]
