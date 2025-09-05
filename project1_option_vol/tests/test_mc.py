# tests/test_mc.py
import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\utkarsh\OneDrive\Desktop\quant-projects\project1_option_vol\src"))
from mc_pricers import mc_arithmetic_asian
import numpy as np

def test_mc_antithetic_reduces_variance():
    S0 = 100.0; K = 100.0; r = 0.01; sigma = 0.25; T = 0.5
    # run two small MCs with same seed to compare sample std dev
    price1, stderr1 = mc_arithmetic_asian(S0, K, r, sigma, T, M=50, N=2000, option="call", antithetic=False, seed=42)
    price2, stderr2 = mc_arithmetic_asian(S0, K, r, sigma, T, M=50, N=2000, option="call", antithetic=True, seed=42)
    # antithetic should produce lower or equal stderr in expectation
    assert stderr2 <= stderr1 * 1.1  # allow tiny numerical wiggle
