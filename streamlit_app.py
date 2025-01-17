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
SUMMER_MONTHS = [5, 6, 7, 8]  # June to September
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]  # January to May, October to December

# Helper Functions
def calculate_monthly_values(daily_value):
    return [daily_value * days for days in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    daily_demand = miles / efficiency
    total_days = days_per_week * 52
    yearly_demand = daily_demand * total_days
    monthly_demand = calculate_monthly_values(daily_demand * days_per_week / 7)
    return yearly_demand, monthly_demand

def calculate_solar_production(size):
    yearly_production = size * 4 * 365  # Assume 4 kWh/day per kW system
    monthly_production = calculate_monthly_values(size * 4)
    return yearly_production, monthly_production

def calculate_monthly_costs(ev_monthly, solar_monthly, household_monthly, battery_capacity, time_of_charging):
    ev_cost_no_solar = []
    ev_cost_nem_2 = []
    nem_3_battery_costs = []
    total_cost_no_solar = []
    total_cost_nem_2 = []
    total_cost_nem_3 = []

    battery_state = 0  # Start with an empty battery

    for month in range(12):
        rates = TOU_RATES["summer"] if month in SUMMER_MONTHS else TOU_RATES["winter"]

        # Household costs (no solar)
        household_cost_no_solar = household_monthly[month] * rates["off_peak"]

        # EV cost (no solar, charged at night)
        ev_cost = ev_monthly[month] * rates["super_off_peak"]
        ev_cost_no_solar.append(ev_cost)

        # Total cost without solar
        total_cost_no_solar.append(household_cost_no_solar + ev_cost)

        # EV cost under NEM 2.0
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit_nem_2 = max(0, excess_solar * rates["off_peak"])
        ev_cost_under_nem_2 = max(0, ev_monthly[month] - credit_nem_2)
        ev_cost_nem_2.append(ev_cost_under_nem_2)
        total_cost_nem_2.append(household_cost_no_solar - credit_nem_2 + ev_cost_under_nem_2)

        # EV cost under NEM 3.0 + Battery
        excess_solar = max(0, solar_monthly[month] - household_monthly[month])

        # Daytime charging: Use direct solar, then battery, then grid at peak rates
        # Nighttime charging: Use battery first, then grid at super off-peak rates
        ev_shortfall = ev_monthly[month]

        if time_of_charging == "Daytime (Peak)" and excess_solar > 0:
            direct_solar_used = min(ev_shortfall, excess_solar)
            ev_shortfall -= direct_solar_used
            excess_solar -= direct_solar_used

        # Charge battery with remaining solar
        if excess_solar > 0 and battery_state < battery_capacity:
            battery_charge = min(excess_solar, battery_capacity - battery_state)
            battery_state += battery_charge * DEFAULT_BATTERY_EFFICIENCY
            excess_solar -= battery_charge

        # Discharge battery to meet EV demand
        if ev_shortfall > 0 and battery_state > 0:
            battery_discharge = min(ev_shortfall, battery_state)
            ev_shortfall -= battery_discharge
            battery_state -= battery_discharge

        # Remaining EV demand is met by the grid
        if time_of_charging == "Night (Super Off-Peak)":
            grid_energy_used = ev_shortfall
            nem_3_cost = grid_energy_used * rates["super_off_peak"]
        elif time_of_charging == "Daytime (Peak)":
            grid_energy_used = ev_shortfall
            nem_3_cost = grid_energy_used * rates["on_peak"]

        nem_3_battery_costs.append(nem_3_cost)
        total_cost_nem_3.append(household_cost_no_solar + nem_3_cost)

    return ev_cost_no_solar, ev_cost_nem_2, nem_3_battery_costs, total_cost_no_solar, total_cost_nem_2, total_cost_nem_3

# Streamlit App
st.title("Energy Simulation Dashboard")

# Sidebar Parameters
st.sidebar.header("Simulation Parameters")

# EV Parameters
with st.sidebar.expander("EV Parameters"):
    commute_miles = st.slider("Daily Commute Distance (miles)", 10, 100, int(DEFAULT_COMMUTE_MILES), step=1)
    ev_model = st.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days = st.radio("Charging Frequency", ["Daily", "Weekdays Only"])
    time_of_charging = st.radio(
        "Time of Charging",
        ["Night (Super Off-Peak)", "Daytime (Peak)"],
        index=0,
    )
    ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, 5 if charging_days == "Weekdays Only" else 7)

# Household Consumption
with st.sidebar.expander("Household Consumption"):
    household_consumption = st.slider("Average Daily Consumption (kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION), step=1)
    fluctuation = st.slider("Consumption Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION * 100), step=1) / 100
    household_yearly = household_consumption * (1 + fluctuation) * 365
    household_monthly = calculate_monthly_values(household_consumption * (1 + fluctuation))

# Solar Production
with st.sidebar.expander("Solar Panel Production"):
    solar_size = st.slider("Solar System Size (kW)", 3, 15, int(DEFAULT_SOLAR_SIZE), step=1)
    battery_capacity = st.slider("Battery Capacity (kWh)", 0, 20, int(DEFAULT_BATTERY_CAPACITY), step=1)
    solar_yearly, solar_monthly = calculate_solar_production(solar_size)

# Calculate Monthly Costs
ev_cost_no_solar, ev_cost_nem_2, nem_3_battery_costs, total_cost_no_solar, total_cost_nem_2, total_cost_nem_3 = calculate_monthly_costs(
    ev_monthly, solar_monthly, household_monthly, battery_capacity, time_of_charging
)

# Monthly Results
monthly_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "EV Consumption (kWh)": ev_monthly,
    "Household Consumption (kWh)": household_monthly,
    "Total Consumption (EV + Household, kWh)": [ev + hh for ev, hh in zip(ev_monthly, household_monthly)],
    "Solar Production (kWh)": solar_monthly,
    "EV Charging Cost (No Solar, $)": ev_cost_no_solar,
    "EV Charging Cost (Solar with NEM 2.0, $)": ev_cost_nem_2,
    "EV Charging Cost (Solar with NEM 3.0 + Battery, $)": nem_3_battery_costs,
    "Total Cost (No Solar, EV + Household, $)": total_cost_no_solar,
    "Total Cost (Solar with NEM 2.0 + EV + Household, $)": total_cost_nem_2,
    "Total Cost (Solar with NEM 3.0 + Battery + EV + Household, $)": total_cost_nem_3,
})

# Results Section
st.header("Simulation Results")
st.write("### Monthly Results Breakdown")
st.table(monthly_data)

# Visualizations
st.write("### Visualizations")

# 1. Chart: Total Consumption
st.write("#### Total Energy Consumption")
fig, ax = plt.subplots()
ax.bar(monthly_data["Month"], monthly_data["EV Consumption (kWh)"], label="EV Consumption")
ax.bar(monthly_data["Month"], monthly_data["Household Consumption (kWh)"], bottom=monthly_data["EV Consumption (kWh)"], label="Household Consumption")
ax.plot(monthly_data["Month"], monthly_data["Total Consumption (EV + Household, kWh)"], label="Total Consumption", color="red", linestyle="--")
ax.set_ylabel("Energy (kWh)")
ax.legend()
st.pyplot(fig)

# 2. Chart: Solar Production
st.write("#### Solar Production")
fig, ax = plt.subplots()
ax.plot(monthly_data["Month"], monthly_data["Solar Production (kWh)"], label="Solar Production", color="gold", marker="o")
ax.set_ylabel("Energy (kWh)")
ax.legend()
st.pyplot(fig)

# 3. Chart: EV Charging Costs
st.write("#### EV Charging Costs")
fig, ax = plt.subplots()
ax.plot(monthly_data["Month"], monthly_data["EV Charging Cost (No Solar, $)"], label="No Solar")
ax.plot(monthly_data["Month"], monthly_data["EV Charging Cost (Solar with NEM 2.0, $)"], label="NEM 2.0")
ax.plot(monthly_data["Month"], monthly_data["EV Charging Cost (Solar with NEM 3.0 + Battery, $)"], label="NEM 3.0 + Battery")
ax.set_ylabel("Cost ($)")
ax.legend()
st.pyplot(fig)

# 4. Chart: Total Costs to User
st.write("#### Total Monthly Costs")
fig, ax = plt.subplots()
ax.plot(monthly_data["Month"], monthly_data["Total Cost (No Solar, EV + Household, $)"], label="No Solar")
ax.plot(monthly_data["Month"], monthly_data["Total Cost (Solar with NEM 2.0 + EV + Household, $)"], label="Solar + NEM 2.0")
ax.plot(monthly_data["Month"], monthly_data["Total Cost (Solar with NEM 3.0 + Battery + EV + Household, $)"], label="Solar + NEM 3.0 + Battery")
ax.set_ylabel("Cost ($)")
ax.legend()
st.pyplot(fig)
