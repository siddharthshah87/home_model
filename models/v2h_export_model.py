# models/v2h_export_model.py

import numpy as np

# Simulate dynamic use of EV battery for grid replacement or TOU arbitrage

def simulate_v2h_savings(ev_batt_kwh,
                         max_discharge_kw=7.2,
                         efficiency=0.9,
                         days_active=150,
                         discharge_hours_per_day=3,
                         peak_rate=0.45,
                         offpeak_rate=0.12,
                         smart_panel=None):
    """
    Simulates V2H savings assuming fixed discharge window from EV battery
    into home during high-rate hours.
    Returns estimated annual dollar savings.
    """
    if not smart_panel or not smart_panel.is_load_controlled("ev"):
        return 0.0

    daily_energy_export_kwh = min(ev_batt_kwh * efficiency, max_discharge_kw * discharge_hours_per_day)
    rate_diff = peak_rate - offpeak_rate

    annual_savings = daily_energy_export_kwh * days_active * rate_diff
    return round(annual_savings, 2)

