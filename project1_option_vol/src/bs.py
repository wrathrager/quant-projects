# Black-Scholes price + Greeks
# src/bs.py
import numpy as np
from scipy.stats import norm

def d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, q, sigma, T):
    return d1(S,K,r,q,sigma,T) - sigma*np.sqrt(T)

def bs_price(S, K, r, q, sigma, T, option='call'):
    if T <= 0:
        if option=='call':
            return max(S-K, 0.0)
        return max(K-S, 0.0)
    D1 = d1(S,K,r,q,sigma,T)
    D2 = D1 - sigma*np.sqrt(T)
    df = np.exp(-r*T)
    if option == 'call':
        return np.exp(-q*T)*S*norm.cdf(D1) - df*K*norm.cdf(D2)
    else:
        return df*K*norm.cdf(-D2) - np.exp(-q*T)*S*norm.cdf(-D1)

def bs_greeks(S, K, r, q, sigma, T):
    D1 = d1(S,K,r,q,sigma,T)
    D2 = D1 - sigma*np.sqrt(T)
    pdf = norm.pdf(D1)
    delta_call = np.exp(-q*T)*norm.cdf(D1)
    gamma = np.exp(-q*T)*pdf/(S*sigma*np.sqrt(T))
    vega = S*np.exp(-q*T)*pdf*np.sqrt(T)
    theta = (-S*pdf*sigma*np.exp(-q*T)/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(D2) + q*S*np.exp(-q*T)*norm.cdf(D1))
    rho = K*T*np.exp(-r*T)*norm.cdf(D2)
    return {'delta_call': delta_call, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}
