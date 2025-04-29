# models/basic_hourly_model.py (UPDATED for Smart Panel Awareness)

import numpy as np
import pandas as pd

HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16,21)),
    "off_peak_hours": list(range(7,16)) + [21,22],
    "super_off_peak_hours": list(range(0,7)) + [23]
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90


def classify_tou_basic(hour):
    if hour in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"


def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh,
                        battery_state, battery_capacity, smart_panel=None):
    hour_of_day = hour_idx % 24
    period = classify_tou_basic(hour_of_day)
    rate = HOUR_TOU_RATES_BASIC[period]

    total_demand = house_kwh + ev_kwh

    # SMART PANEL CONTROL
    if smart_panel and smart_panel.panel_type != "Legacy":
        # Defer EV charging during peak hours if controllable
        if smart_panel.is_load_controlled("ev") and period == "on_peak":
            ev_kwh = 0.0
            total_demand = house_kwh

    if solar_kwh >= total_demand:
        leftover_solar = solar_kwh - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_kwh

    solar_unused = 0
    if leftover_solar > 0 and battery_state < battery_capacity:
        available_space = (battery_capacity - battery_state) / BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery = min(leftover_solar, available_space)
        battery_state += to_battery * BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar -= to_battery
        solar_unused = leftover_solar
    else:
        solar_unused = leftover_solar

    if total_demand > 0 and battery_state > 0:
        discharge = min(total_demand, battery_state)
        total_demand -= discharge
        battery_state -= discharge

    grid_kwh = total_demand
    cost = grid_kwh * rate
    return battery_state, grid_kwh, cost, solar_unused


def build_basic_ev_shape(pattern="Night"):
    shape = np.zeros(24)
    if pattern == "Night":
        hrs_night = list(range(18,24)) + list(range(0,7))
        for h in hrs_night:
            shape[h] = 1
        shape /= shape.sum()
    else:
        hrs_day = list(range(9,17))
        for h in hrs_day:
            shape[h] = 1
        shape /= shape.sum()
    return shape


def run_basic_hourly_sim(daily_house, daily_solar, daily_ev,
                         battery_capacity=10.0,
                         ev_charging_pattern="Night",
                         reset_battery_daily=False,
                         smart_panel=None):
    ev_shape = build_basic_ev_shape(ev_charging_pattern)
    house_shape = np.array([
        0.02,0.02,0.02,0.02,0.02,0.03,0.04,0.06,
        0.06,0.06,0.06,0.05,0.05,0.05,0.06,0.07,
        0.08,0.06,0.05,0.05,0.04,0.03,0.02,0.02
    ])
    house_shape /= house_shape.sum()

    solar_shape = np.array([
        0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.10,
        0.15,0.20,0.20,0.15,0.10,0.05,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()

    results = {
        "day": [], "hour": [], "house_kwh": [], "ev_kwh": [], "solar_kwh": [],
        "grid_kwh": [], "cost": [], "battery_state": [], "solar_unused": []
    }

    battery_state = 0.0
    total_cost = 0.0
    total_grid = 0.0
    total_solar_unused = 0.0

    days = len(daily_house)
    for d in range(days):
        if reset_battery_daily:
            battery_state = 0.0

        dh = daily_house[d]
        ds = daily_solar[d]
        de = daily_ev[d]

        h_24 = dh * house_shape
        s_24 = ds * solar_shape
        e_24 = de * ev_shape

        for hour in range(24):
            hour_idx = d*24 + hour
            bh = h_24[hour]
            bs = s_24[hour]
            be = e_24[hour]

            battery_state, g_kwh, cost, sol_un = simulate_hour_basic(
                hour_idx, bs, bh, be, battery_state, battery_capacity, smart_panel
            )
            total_cost += cost
            total_grid += g_kwh
            total_solar_unused += sol_un

            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(bh)
            results["ev_kwh"].append(be)
            results["solar_kwh"].append(bs)
            results["grid_kwh"].append(g_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(sol_un)

    df = pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df
