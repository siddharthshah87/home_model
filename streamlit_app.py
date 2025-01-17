import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10  # kWh
DEFAULT_BATTERY_EFFICIENCY = 0.9  # 90%
DEFAULT_SOLAR_SIZE = 7.5  # kW
TOU_RATES = {
    "summer": {"on_peak": 0.45, "off_peak": 0.25, "super_off_peak": 0.12},
    "winter": {"on_peak": 0.35, "off_peak": 0.20, "super_off_peak": 0.10},
}
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8  # kWh/day
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2  # 20%
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
CHARGING_POWER = 11.52  # kW for 48A EVSE at 240V
SUMMER_MONTHS = [5, 6, 7, 8]  # June to September
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]  # January to May, October to December

# Helper Functions
def calculate_monthly_values(daily_value):
    """Distribute daily value into monthly totals."""
    return [daily_value * days for days in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    """Calculate annual EV energy demand and monthly breakdown."""
    daily_demand = miles / efficiency
    total_days = days_per_week * 52
    yearly_demand = daily_demand * total_days
    monthly_demand = calculate_monthly_values(daily_demand * days_per_week / 7)
    return yearly_demand, monthly_demand

def calculate_solar_production(size):
    """Simulate yearly and monthly solar energy production."""
    yearly_production = size * 4 * 365  # Assume 4 kWh/day per kW system
    monthly_production = calculate_monthly_values(size * 4)
    return yearly_production, monthly_production

def calculate_charging_time(daily_ev_demand, charging_power=CHARGING_POWER, charging_efficiency=0.9):
    """Calculate time required to charge the EV daily."""
    net_energy = daily_ev_demand / charging_efficiency
    charging_time = net_energy / charging_power
    return charging_time

def calculate_monthly_ev_costs(ev_monthly, rates, solar_monthly, nem_plan, battery_capacity):
    """Calculate monthly EV costs under various scenarios."""
    ev_cost_nem_2 = []
    ev_cost_no_solar = []
    ev_cost_nem_3 = []
    total_cost_per_month = []

    for month in range(12):
        # Assign seasonal rates
        seasonal_rates = TOU_RATES["summer"] if month in SUMMER_MONTHS else TOU_RATES["winter"]

        # EV cost with no solar (grid-only charging)
        cost_no_solar = ev_monthly[month] * seasonal_rates["super_off_peak"]
        ev_cost_no_solar.append(cost_no_solar)

        # EV cost under NEM 2.0
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit = max(0, excess_solar * seasonal_rates["off_peak"])
        cost_nem_2 = max(0, ev_monthly[month] - credit)
        ev_cost_nem_2.append(cost_nem_2)

        # EV cost under NEM 3.0 (self-consumption priority)
        battery_state = 0
        if battery_capacity > 0:
            charge = min(excess_solar, battery_capacity - battery_state)
            battery_state += charge * DEFAULT_BATTERY_EFFICIENCY
            excess_solar -= charge

        ev_shortfall = ev_monthly[month]
        if battery_state > 0:
            discharge = min(ev_shortfall, battery_state)
            battery_state -= discharge
            ev_shortfall -= discharge

        cost_nem_3 = ev_shortfall * seasonal_rates["super_off_peak"]
        ev_cost_nem_3.append(cost_nem_3)

        # Total monthly cost
        household_cost = household_monthly[month] * seasonal_rates["off_peak"]
        total_cost = household_cost + cost_nem_3
        total_cost_per_month.append(total_cost)

    return ev_cost_nem_2, ev_cost_no_solar, ev_cost_nem_3, total_cost_per_month

# Streamlit App
st.title("Energy Simulation Dashboard")

# Tabs
st.sidebar.header("Simulation Parameters")

# Tab 1: EV Parameters
with st.sidebar.expander("EV Parameters"):
    commute_miles = st.slider("Daily Commute Distance (miles)", 10, 100, int(DEFAULT_COMMUTE_MILES), step=1)
    ev_model = st.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days = st.radio("Charging Frequency", ["Daily", "Weekdays Only"])
    ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, 5 if charging_days == "Weekdays Only" else 7)
    charging_time = calculate_charging_time(ev_yearly / 365)

    st.write(f"**Daily EV Energy Demand**: {ev_yearly / 365:.2f} kWh")
    st.write(f"**Time to Charge (at 48A, 240V)**: {charging_time:.2f} hours")

# Tab 2: Utility Rates
with st.sidebar.expander("Utility Rates"):
    nem_plan = st.radio("NEM Plan", ["NEM 2.0", "NEM 3.0"])

# Tab 3: Household Consumption
with st.sidebar.expander("Household Consumption"):
    household_consumption = st.slider("Average Daily Consumption (kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION), step=1)
    fluctuation = st.slider("Consumption Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION * 100), step=1) / 100
    household_yearly = household_consumption * (1 + fluctuation) * 365
    household_monthly = calculate_monthly_values(household_consumption * (1 + fluctuation))

# Tab 4: Solar Panel Production
with st.sidebar.expander("Solar Panel Production"):
    solar_size = st.slider("Solar System Size (kW)", 3, 15, int(DEFAULT_SOLAR_SIZE), step=1)
    battery_capacity = st.slider("Battery Capacity (kWh)", 0, 20, int(DEFAULT_BATTERY_CAPACITY), step=1)
    solar_yearly, solar_monthly = calculate_solar_production(solar_size)

# Calculate Monthly EV Costs
ev_cost_nem_2, ev_cost_no_solar, ev_cost_nem_3, total_cost_per_month = calculate_monthly_ev_costs(
    ev_monthly, TOU_RATES, solar_monthly, nem_plan, battery_capacity
)

# Monthly Results
monthly_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "EV Consumption (kWh)": ev_monthly,
    "Household Consumption (kWh)": household_monthly,
    "Solar Production (kWh)": solar_monthly,
    "Household Monthly Bill ($)": [household_monthly[month] * (TOU_RATES["summer" if month in SUMMER_MONTHS else "winter"]["off_peak"]) for month in range(12)],
    "EV Monthly Cost (NEM 2.0, $)": ev_cost_nem_2,
    "EV Monthly Cost (No Solar, $)": ev_cost_no_solar,
    "EV Monthly Cost (NEM 3.0, $)": ev_cost_nem_3,
    "Total Monthly Cost ($)": total_cost_per_month,
})

# Results Section
st.header("Simulation Results")

# Display the table
st.write("### Monthly Results Breakdown")
st.table(monthly_data)
