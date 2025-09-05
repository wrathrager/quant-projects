# tests/test_binomial.py
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\utkarsh\OneDrive\Desktop\quant-projects\project1_option_vol\src"))
import math
from binomial import crr_price
from bs import black_scholes_price

def test_binomial_converges_to_bs():
    S = 100.0
    K = 100.0
    r = 0.01
    q = 0.0
    sigma = 0.2
    T = 1.0
    option = "call"

    bs = black_scholes_price(S, K, T, r, sigma, option_type=option)

    price_small = crr_price(S, K, r, q, sigma, T, N=10, option=option, american=False)
    price_med   = crr_price(S, K, r, q, sigma, T, N=200, option=option, american=False)
    price_big   = crr_price(S, K, r, q, sigma, T, N=800, option=option, american=False)

    # Binomial with larger N should be closer to BS
    err_small = abs(price_small - bs)
    err_med = abs(price_med - bs)
    err_big = abs(price_big - bs)

    assert err_med < err_small + 1e-9
    assert err_big < err_med + 1e-9
