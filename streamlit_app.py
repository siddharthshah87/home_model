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
    # === User Utility Plan Info ===
    st.sidebar.header("User & Utility Info")
    user_state = st.sidebar.selectbox("Select Your State", [
        "AL","AR","AZ","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN",
        "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE",
        "NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX",
        "UT","VA","VT","WA","WI","WV","WY"
    ])

    user_info = {"state": user_state}

    if st.sidebar.button("🔎 Fetch Rate Plans"):
        with st.spinner("Retrieving plans for your state..."):
            plans = fetch_urdb_plans_for_state(user_info["state"])
            if plans:
                st.session_state["urdb_plans"] = plans
                st.sidebar.success(f"Found {len(plans)} plans for {user_info['state']}")
            else:
                st.sidebar.warning("No plans found or error.")

    # Show plans if available
    chosen_plan = None
    if "urdb_plans" in st.session_state:
        plan_names = [p.get("name", "Unnamed") for p in st.session_state["urdb_plans"] if p.get("sector") == "Residential"]
        if plan_names:
            selected = st.sidebar.selectbox("Select a Residential Rate Plan", plan_names)
            chosen_plan = next((p for p in st.session_state["urdb_plans"] if p.get("name") == selected), None)
            if chosen_plan:
                st.sidebar.markdown("---")
                st.sidebar.markdown(f"**Selected Plan:**{chosen_plan.get('name')}")
                parsed = naive_parse_urdb_plan(chosen_plan)
                st.session_state["urdb_rate_structure"] = parsed
                st.sidebar.success(f"Parsed Rate: {parsed}")
        else:
            st.sidebar.info("No residential plans found.")

    st.sidebar.markdown("---")

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

# === Main Page UI ===
    st.title("🔌 Smart Panel Energy Dashboard")

    st.markdown("---")

    st.subheader("Configuration Summary")
    load_profile = {
        "panel_type": panel_type,
        "ev": {
            "enabled": control_ev,
            "battery_kwh": ev_battery_kwh,
            "charging_pattern": ev_charging_pattern,
            "charging_days": ev_days_active,
        },
        "hvac": {
            "enabled": control_hvac,
            "runtime_hrs": hvac_runtime_hours,
            "load_kw": hvac_kw,
        },
        "water_heater": {
            "enabled": control_water_heater,
            "load_kw": water_heater_kw,
        },
        "washer_dryer": {
            "enabled": control_washer_dryer,
            "runtime_hrs": washer_runtime_hrs,
            "load_kw": washer_kw,
        },
        "dishwasher": {
            "enabled": control_dishwasher,
            "cycles_week": dishwasher_cycles,
            "load_kw": dishwasher_kw,
        },
        "pool_pump": {
            "enabled": control_pool_pump,
            "runtime_hrs": pool_runtime,
            "load_kw": pool_kw,
        },
        "battery": {
            "enabled": control_battery,
            "capacity_kwh": home_battery_kwh,
        },
        "solar": {
            "enabled": control_solar_inverter,
            "size_kw": solar_size,
            "orientation": solar_orientation,
            "shading": solar_shading,
        }
    }

    st.json(load_profile)

    st.markdown("---")

    # Simulation trigger
    if st.button("🚀 Run Simulation"):
        daily_house_arr = np.full(DAYS_PER_YEAR, DEFAULT_HOUSEHOLD_CONSUMPTION)
        daily_solar_arr = np.full(DAYS_PER_YEAR, load_profile["solar"]["size_kw"] * 4)
        daily_ev_arr = np.full(DAYS_PER_YEAR, (DEFAULT_COMMUTE_MILES / 4.0) * (load_profile["ev"]["charging_days"] / 7.0))

        cost, grid_kwh, unused_solar, df_results = run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            load_profile["battery"]["capacity_kwh"],
            load_profile["ev"]["charging_pattern"],
            reset_battery_daily=False,
            smart_panel=smart_panel
        )
        st.success("Simulation complete! Sample output below.")

        # --- Real Key Metrics ---
        st.subheader("📊 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annual Energy Cost", f"${cost:,.0f}")
        col2.metric("V2H Savings", "$0 (N/A)")
        col3.metric("Payback Period", "N/A")
        col4.metric("Resilience Score", "Pending")

        st.markdown("---")
        
        # --- Sample Charts Placeholder ---
        st.subheader("📈 Visualization")
        chart_type = st.radio("Select View", ["Cost Comparison", "Hourly Load", "Resilience Profile"])

        if chart_type == "Cost Comparison":
            st.bar_chart(pd.DataFrame({"Legacy Panel": [1850], "Smart Panel": [1523]}))
        elif chart_type == "Hourly Load":
            st.line_chart(np.random.rand(24))
        elif chart_type == "Resilience Profile":
            st.area_chart(np.random.rand(24))

        st.markdown("---")

        # --- Insights Block ---
        st.subheader("💡 Smart Recommendations")
        st.success("Switching EV charging to night hours saves $210/year.")
        st.warning("Washer/Dryer overlaps with peak hours. Consider deferral.")

        st.markdown("---")
        
        # --- Actions ---
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🔄 Reset Inputs"):
                st.experimental_rerun()
        with colB:
            if st.button("🖨️ Export Summary"):
                st.info("Feature coming soon.")

if __name__=="__main__":
    main()
