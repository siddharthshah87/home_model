# models/v2h_export_model.py (New)

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


# models/load_priority_model.py (New)

CRITICAL_LOADS = ["refrigerator", "lighting", "wifi", "garage_opener"]
FLEXIBLE_LOADS = ["EV", "HVAC", "WasherDryer", "WaterHeater"]


def classify_load_priority(load_name):
    if load_name in CRITICAL_LOADS:
        return "critical"
    elif load_name in FLEXIBLE_LOADS:
        return "flexible"
    return "unknown"


def summarize_load_portfolio(smart_panel):
    active = smart_panel.get_controlled_loads()
    crit = [l for l in active if classify_load_priority(l) == "critical"]
    flex = [l for l in active if classify_load_priority(l) == "flexible"]
    return {
        "critical": crit,
        "flexible": flex,
        "total_managed": len(active)
    }


# models/installer_finance_model.py (New)

def estimate_payback_costs(panel_type, smart_features=True,
                            panel_upgrade_avoided=True,
                            v2h_enabled=True,
                            base_cost=2500,
                            install_cost=1200,
                            panel_upgrade_cost=4000):
    """
    Simple financial model to estimate effective system cost and payback.
    """
    total_cost = base_cost + install_cost

    if panel_upgrade_avoided:
        total_cost -= panel_upgrade_cost

    if not smart_features:
        total_cost -= 500  # no load control savings

    if not v2h_enabled:
        total_cost -= 750  # no inverter/V2H interface

    return max(total_cost, 0)
