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
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Helper Functions
def calculate_monthly_values(daily_value):
    """Distribute daily value into monthly totals."""
    monthly_values = [daily_value * days for days in DAYS_IN_MONTH]
    return monthly_values

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

def simulate_costs(household, solar, ev_demand, battery_capacity, rates, nem_plan):
    """Simulate energy flows and costs (yearly and monthly)."""
    yearly_cost = 0
    monthly_cost = []
    grid_import = []
    grid_export = []
    
    for month, days in enumerate(DAYS_IN_MONTH):
        monthly_household = household / 12
        monthly_solar = solar / 12
        monthly_ev = ev_demand[month]

        # Monthly energy flows
        excess_solar = max(monthly_solar - monthly_household, 0)
        battery_state = 0  # Reset battery monthly
        if battery_capacity > 0:
            battery_state = min(excess_solar, battery_capacity)

        grid_import_month = max(monthly_household + monthly_ev - monthly_solar - battery_state, 0)
        grid_export_month = max(monthly_solar - (monthly_household + monthly_ev), 0)

        # Monthly costs
        monthly_rate = rates["super_off_peak"]
        monthly_cost.append(grid_import_month * monthly_rate)
        grid_import.append(grid_import_month)
        grid_export.append(grid_export_month)

        yearly_cost += grid_import_month * monthly_rate

    return {
        "Yearly Cost ($)": yearly_cost,
        "Monthly Cost ($)": monthly_cost,
        "Grid Import (kWh)": grid_import,
        "Grid Export (kWh)": grid_export,
    }

# Streamlit App
st.title("Energy Simulation Dashboard with Monthly Analysis")

# Tabs
st.sidebar.header("Simulation Parameters")

# Tab 1: EV Parameters
with st.sidebar.expander("EV Parameters"):
    commute_miles = st.slider("Daily Commute Distance (miles)", 
                          min_value=10, 
                          max_value=100, 
                          value=int(DEFAULT_COMMUTE_MILES), 
                          step=1)
    ev_model = st.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days = st.radio("Charging Frequency", ["Daily", "Weekdays Only"])
    ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, 5 if charging_days == "Weekdays Only" else 7)

# Tab 2: Utility Rates
with st.sidebar.expander("Utility Rates"):
    tou_plan = st.radio("Time-of-Use Plan", ["Summer", "Winter"])
    rates = TOU_RATES[tou_plan.lower()]
    nem_plan = st.radio("NEM Plan", ["NEM 2.0", "NEM 3.0"])

# Tab 3: Household Consumption
with st.sidebar.expander("Household Consumption"):
    household_consumption = st.slider("Average Daily Consumption (kWh)", 
                                  min_value=10, 
                                  max_value=50, 
                                  value=int(DEFAULT_HOUSEHOLD_CONSUMPTION), 
                                  step=1)  # Set step to 1 (integer)
    fluctuation = st.slider("Consumption Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION * 100)) / 100
    household_yearly = household_consumption * (1 + fluctuation) * 365
    household_monthly = calculate_monthly_values(household_consumption * (1 + fluctuation))

# Tab 4: Solar Panel Production
with st.sidebar.expander("Solar Panel Production"):
    solar_size = st.slider("Solar System Size (kW)", 
                       min_value=3, 
                       max_value=15, 
                       value=int(DEFAULT_SOLAR_SIZE), 
                       step=1)
    battery_capacity = st.slider("Battery Capacity (kWh)", 0, 20, DEFAULT_BATTERY_CAPACITY)
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