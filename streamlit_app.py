import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Import from our local modules
from models.monthly_model import (
    calculate_ev_demand, calculate_solar_production, calculate_monthly_values,
    calculate_monthly_costs
)
from models.basic_hourly_model import (
    run_basic_hourly_sim
)
from models.advanced_lp_model import (
    run_advanced_lp_sim
)
from utils.param_sweep import (
    param_sweep_battery_sizes
)
from services.urdb_api import (
    fetch_urdb_plans_for_state, naive_parse_urdb_plan
)
from services.bill_parsing import (
    parse_utility_bill_pdf
)
from services.recommendation import (
    get_deepseek_recommendations
)

# Constants shared in app
DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]
DAYS_PER_YEAR = 365
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DEFAULT_COMMUTE_MILES = 30
DEFAULT_SOLAR_SIZE = 7.5
DEFAULT_BATTERY_CAPACITY = 10
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2

def main():
    # === Smart Panel Setup ===
    st.sidebar.header("Smart Panel Setup")

    panel_type = st.sidebar.selectbox(
        "1. What kind of panel do you have?",
        ["Legacy", "Smart Subpanel", "Smart Full Panel"]
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("2. Select Which Loads Are Connected to the Smart Panel")
    control_ev = st.sidebar.checkbox("EV Charger", value=True)
    control_hvac = st.sidebar.checkbox("HVAC System", value=False)
    control_water_heater = st.sidebar.checkbox("Water Heater", value=False)
    control_solar_inverter = st.sidebar.checkbox("Solar Inverter", value=False)
    control_battery = st.sidebar.checkbox("Home Battery", value=False)
    control_washer_dryer = st.sidebar.checkbox("Washer/Dryer", value=False)
    control_dishwasher = st.sidebar.checkbox("Dishwasher", value=False)
    control_pool_pump = st.sidebar.checkbox("Pool Pump", value=False)

    # Create a SmartPanelConfig object
    from models.panel_model import SmartPanelConfig

    smart_panel = SmartPanelConfig(
        panel_type=panel_type,
        control_ev=control_ev,
        control_hvac=control_hvac,
        control_water_heater=control_water_heater,
        control_solar_inverter=control_solar_inverter,
        control_battery=control_battery
    )

    st.sidebar.markdown("---")

    # 3. EV-Specific Inputs
    if control_ev:
        st.sidebar.subheader("3. EV Configuration")
        charging_freq = st.sidebar.radio("EV Charging Frequency", ["Daily", "Weekdays Only"])
        default_ev_days_per_week = 5 if charging_freq == "Weekdays Only" else 7

        ev_battery_kwh = st.sidebar.slider("EV Battery Size (kWh)", 20, 150, 50)
        ev_charging_pattern = st.sidebar.radio("Preferred Charging Time", ["Night", "Daytime"])
        ev_days_active = st.sidebar.slider("Charging Days per Week", 1, 7, default_ev_days_per_week)
    else:
        ev_battery_kwh = 0
        ev_charging_pattern = "Night"
        ev_days_active = 0

    # 4. HVAC-Specific Inputs
    if control_hvac:
        st.sidebar.subheader("4. HVAC Configuration")
        hvac_runtime_hours = st.sidebar.slider("HVAC Runtime per Day (hrs)", 2, 24, 8)
        hvac_kw = st.sidebar.slider("HVAC Load (kW)", 1.0, 10.0, 3.5)
    else:
        hvac_runtime_hours = 0
        hvac_kw = 0

    # 5. Water Heater Inputs
    if control_water_heater:
        st.sidebar.subheader("5. Water Heater Configuration")
        water_heater_kw = st.sidebar.slider("Water Heater Load (kW)", 1.0, 8.0, 4.5)
    else:
        water_heater_kw = 0

    # 6. Washer/Dryer Inputs
    if control_washer_dryer:
        st.sidebar.subheader("6. Washer/Dryer Configuration")
        washer_runtime_hrs = st.sidebar.slider("Washer/Dryer Runtime (hrs/day)", 0, 3, 1)
        washer_kw = st.sidebar.slider("Washer/Dryer Load (kW)", 0.5, 3.0, 2.0)
    else:
        washer_runtime_hrs = 0
        washer_kw = 0

    # 7. Dishwasher Inputs
    if control_dishwasher:
        st.sidebar.subheader("7. Dishwasher Configuration")
        dishwasher_cycles = st.sidebar.slider("Dishwasher Cycles per Week", 0, 14, 5)
        dishwasher_kw = st.sidebar.slider("Dishwasher Load (kW)", 0.5, 2.0, 1.2)
    else:
        dishwasher_cycles = 0
        dishwasher_kw = 0

    # 8. Pool Pump Inputs
    if control_pool_pump:
        st.sidebar.subheader("8. Pool Pump Configuration")
        pool_runtime = st.sidebar.slider("Pump Runtime (hrs/day)", 0, 24, 4)
        pool_kw = st.sidebar.slider("Pump Load (kW)", 0.5, 3.0, 1.5)
    else:
        pool_runtime = 0
        pool_kw = 0

    # 9. Battery System Inputs
    if control_battery:
        st.sidebar.subheader("9. Home Battery Configuration")
        home_battery_kwh = st.sidebar.slider("Home Battery Size (kWh)", 5, 40, 10)
    else:
        home_battery_kwh = 0

    # 10. Solar Configuration
    if control_solar_inverter:
        st.sidebar.markdown("---")
        st.sidebar.subheader("10. Solar Configuration")
        solar_size = st.sidebar.slider("Solar PV Size (kW)", 0.0, 20.0, float(DEFAULT_SOLAR_SIZE))
        solar_orientation = st.sidebar.radio("Orientation", ["South", "East-West", "Flat"])
        solar_shading = st.sidebar.radio("Shading Level", ["None", "Light", "Moderate", "Heavy"])
    else:
        solar_shading = 0
        solar_orientation = 0
        solar_size = 0

if __name__=="__main__":
    main()
