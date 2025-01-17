import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
DEFAULT_COMMUTE_MILES = 30  # Integer
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}  # Float values for efficiency
DEFAULT_BATTERY_CAPACITY = 10  # Integer, kWh
DEFAULT_BATTERY_EFFICIENCY = 0.9  # Float, 90%
DEFAULT_SOLAR_SIZE = 7.5  # Float, kW
TOU_RATES = {
    "summer": {"on_peak": 0.45, "off_peak": 0.25, "super_off_peak": 0.12},  # Floats
    "winter": {"on_peak": 0.35, "off_peak": 0.20, "super_off_peak": 0.10},  # Floats
}
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8  # Float, kWh/day
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2  # Float, 20%
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Integer days per month
CHARGING_POWER = 11.52  # Float, kW for 48A EVSE at 240V

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

def simulate_costs(household, solar, ev_demand, battery_capacity, rates, nem_plan):
    """Simulate energy flows and costs (yearly and monthly) with NEM 2.0 and NEM 3.0 logic."""
    battery_state = 0
    grid_import = []
    grid_export = []
    monthly_cost = []
    total_cost = 0
    export_rate = 0.10 if nem_plan == "NEM 3.0" else rates["off_peak"]

    for month, days in enumerate(DAYS_IN_MONTH):
        monthly_import = 0
        monthly_export = 0
        monthly_total_cost = 0

        for day in range(days):
            daily_household = household / 365
            daily_solar = solar / 365
            daily_ev = ev_demand[month] / days

            # Excess solar after household consumption
            excess_solar = max(daily_solar - daily_household, 0)

            # Battery logic
            if battery_capacity > 0:
                charge = min(excess_solar, battery_capacity - battery_state)
                battery_state += charge * DEFAULT_BATTERY_EFFICIENCY
                excess_solar -= charge

            # EV charging logic
            ev_shortfall = daily_ev
            if excess_solar > 0:
                ev_direct_charge = min(ev_shortfall, excess_solar)
                ev_shortfall -= ev_direct_charge
                excess_solar -= ev_direct_charge
            if ev_shortfall > 0 and battery_state > 0:
                battery_discharge = min(ev_shortfall, battery_state)
                ev_shortfall -= battery_discharge
                battery_state -= battery_discharge

            # Import remaining EV demand
            if ev_shortfall > 0:
                monthly_import += ev_shortfall
                monthly_total_cost += ev_shortfall * rates["super_off_peak"]

            # Handle remaining solar
            if excess_solar > 0:
                if nem_plan == "NEM 2.0":
                    monthly_export += excess_solar
                elif nem_plan == "NEM 3.0":
                    monthly_export += excess_solar

        # Accumulate monthly results
        grid_import.append(monthly_import)
        grid_export.append(monthly_export)
        monthly_cost.append(monthly_total_cost)
        total_cost += monthly_total_cost

    export_credits = sum(grid_export) * export_rate
    net_cost = total_cost - export_credits

    return {
        "Yearly Cost ($)": net_cost,
        "Monthly Cost ($)": monthly_cost,
        "Grid Import (kWh)": grid_import,
        "Grid Export (kWh)": grid_export,
        "Export Credits ($)": export_credits,
    }

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
    tou_plan = st.radio("Time-of-Use Plan", ["Summer", "Winter"])
    rates = TOU_RATES[tou_plan.lower()]
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

# Simulations
results = simulate_costs(
    household_yearly, sum(solar_monthly), ev_monthly, battery_capacity, rates, nem_plan
)

# Results Section
st.header("Simulation Results")

# Yearly Results
st.write("### Yearly Overview")
st.write(f"**Total Yearly EV Energy Demand**: {ev_yearly:.2f} kWh")
st.write(f"**Total Yearly Household Consumption**: {household_yearly:.2f} kWh")
st.write(f"**Total Yearly Solar Production**: {solar_yearly:.2f} kWh")
st.write(f"**Total Yearly Cost**: ${results['Yearly Cost ($)']:.2f}")

# Monthly Results
st.write("### Monthly Breakdown")
monthly_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "EV Demand (kWh)": ev_monthly,
    "Household Consumption (kWh)": household_monthly,
    "Solar Production (kWh)": solar_monthly,
    "Monthly Cost ($)": results["Monthly Cost ($)"],
})
st.table(monthly_data)
