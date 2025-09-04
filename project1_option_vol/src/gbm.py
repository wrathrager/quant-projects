 # GBM simulator & variance reduction
# src/gbm.py
import numpy as np

def gbm_paths(S0, r, q, sigma, T, N_steps, N_paths, seed=None, antithetic=False):
    rng = np.random.default_rng(seed)
    dt = T / N_steps
    nudt = (r - q - 0.5*sigma**2)*dt
    sigsdt = sigma * np.sqrt(dt)
    if antithetic:
        half = N_paths // 2
        Z = rng.standard_normal(size=(N_steps, half))
        Z = np.concatenate([Z, -Z], axis=1)
    else:
        Z = rng.standard_normal(size=(N_steps, N_paths))
    log_increments = nudt + sigsdt * Z
    log_paths = np.cumsum(log_increments, axis=0)
    S_paths = S0 * np.exp(log_paths)
    # prepend S0 at t=0 if desired
    return np.vstack([np.full((1, S_paths.shape[1]), S0), S_paths])
