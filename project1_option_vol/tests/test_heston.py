import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\utkarsh\OneDrive\Desktop\quant-projects\project1_option_vol\src"))
import numpy as np
from heston import heston_char_func, heston_price


def test_heston_char_func():
    # simple parameter set
    params = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04
    }
    u = 1.0 + 1.0j
    S0, r, T = 100, 0.05, 1.0

    phi = heston_char_func(u, S0, r, T, **params)
    assert isinstance(phi, complex)
    assert not np.isnan(phi.real)
    assert not np.isnan(phi.imag)


def test_heston_price_call():
    # European call parameters
    S0, K, r, T = 100, 100, 0.05, 1.0
    params = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04
    }

    call_price = heston_price(S0, K, r, T, params, option_type="call")
    assert call_price > 0
    assert call_price < S0  # call price must be less than underlying


def test_heston_price_put():
    # European put parameters
    S0, K, r, T = 100, 100, 0.05, 1.0
    params = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04
    }

    put_price = heston_price(S0, K, r, T, params, option_type="put")
    assert put_price > 0
    assert put_price < K * np.exp(-r*T)  # put price must be less than discounted strike
from bs import black_scholes_price

def test_heston_converges_to_bs():
    """
    When sigma -> 0 (vol-of-vol) and v0 = theta = const,
    the Heston model should reduce to Black-Scholes.
    """
    S0, K, r, T = 100, 100, 0.05, 1.0
    const_vol = 0.2  # 20% volatility

    # Heston parameters with sigma -> 0, kappa large (force v0 = theta = const)
    params = {
        "kappa": 10.0,
        "theta": const_vol**2,
        "sigma": 1e-8,
        "rho": 0.0,
        "v0": const_vol**2
    }

    heston_call = heston_price(S0, K, r, T, params, option_type="call")
    bs_call = black_scholes_price(S0, K, T, r, const_vol, option_type="call")

    assert np.isclose(heston_call, bs_call, rtol=1e-2), \
        f"Heston {heston_call} vs BS {bs_call}"
