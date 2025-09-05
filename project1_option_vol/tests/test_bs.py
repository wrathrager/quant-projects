# tests/test_bs.py
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\utkarsh\OneDrive\Desktop\quant-projects\project1_option_vol\src"))
from bs import black_scholes_price

def test_bs_intrinsic_limit():
    S = 100.0
    K = 100.0
    T = 1e-8  # almost zero time to expiry
    r = 0.0
    sigma = 1e-8
    call_price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    assert abs(call_price - max(S-K, 0)) < 1e-6

