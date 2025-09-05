# common utilities (discounting, IO)
# src/utils.py
import datetime
import numpy as np

def year_fraction(start_date, end_date):
    """
    Simple year fraction (actual/365). start_date, end_date are datetime.date objects.
    """
    if not isinstance(start_date, datetime.date):
        start_date = datetime.date.fromisoformat(start_date)
    if not isinstance(end_date, datetime.date):
        end_date = datetime.date.fromisoformat(end_date)
    days = (end_date - start_date).days
    return days / 365.0

def discount_factor(r, T):
    """
    Continuous discount factor exp(-r*T).
    """
    return np.exp(-r * T)
# --- IGNORE ---