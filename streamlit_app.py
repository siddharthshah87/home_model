import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#                               CONSTANTS & GLOBALS
# --------------------------------------------------------------------------------

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
SUMMER_MONTHS = [5, 6, 7, 8]  # June (5) to September (8)
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]  # Jan-May, Oct-Dec

# Hourly (simplified) TOU schedule (same all year):
HOUR_TOU_SCHEDULE = {
    "on_peak_hours": list(range(16, 21)),  # 4 PM - 8 PM
    "off_peak_hours": list(range(7, 16)) + [21, 22],  # 7 AM - 3 PM, 9 PM - 10 PM
    "super_off_peak_hours": list(range(0, 7)) + [23],  # 0 AM - 6 AM, 11 PM
}
HOUR_TOU_RATES = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY = 0.90

# --------------------------------------------------------------------------------
#                               HELPER FUNCTIONS
# --------------------------------------------------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~ MONTHLY MODEL FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_monthly_values(daily_value):
    """Multiply a daily value by the number of days in each month to get monthly values."""
    return [daily_value * days for days in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    """
    Return (yearly_kWh, monthly_kWh_list).
    - miles: daily commute distance
    - efficiency: miles/kWh
    - days_per_week: how many days per week EV is driven/charged
    """
    daily_demand = miles / efficiency
    total_days = days_per_week * 52  # approximate 364 days
    yearly_demand = daily_demand * total_days
    # monthly distribution: daily_demand * (days_per_week/7) for each day in the month
    monthly_demand = calculate_monthly_values(daily_demand * (days_per_week / 7))
    return yearly_demand, monthly_demand

def calculate_solar_production(size):
    """
    Return (yearly_kWh, monthly_kWh_list).
    Very rough assumption: 4 kWh/kW/day
    """
    yearly_production = size * 4 * 365
    monthly_production = calculate_monthly_values(size * 4)
    return yearly_production, monthly_production

def calculate_monthly_costs(ev_monthly, solar_monthly, household_monthly,
                            battery_capacity, time_of_charging):
    """
    Monthly net approach comparing:
      1) No solar
      2) Solar with NEM 2.0
      3) Solar with NEM 3.0 + battery
    """
    ev_cost_no_solar = []
    ev_cost_nem_2 = []
    nem_3_battery_costs = []
    total_cost_no_solar = []
    total_cost_nem_2 = []
    total_cost_nem_3 = []

    battery_state = 0  # We carry it month to month (simple model)

    for month in range(12):
        # Choose rates based on summer/winter
        if month in SUMMER_MONTHS:
            rates = TOU_RATES["summer"]
        else:
            rates = TOU_RATES["winter"]

        # 1) No Solar
        #    - Household at off_peak
        #    - EV at super_off_peak
        household_cost_no_solar = household_monthly[month] * rates["off_peak"]
        ev_cost = ev_monthly[month] * rates["super_off_peak"]
        ev_cost_no_solar.append(ev_cost)
        total_cost_no_solar.append(household_cost_no_solar + ev_cost)

        # 2) Solar + NEM 2.0
        #    - Excess solar credited at off_peak rate
        #    - EV usage net against that credit
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit_nem_2 = max(0, excess_solar * rates["off_peak"])
        ev_cost_under_nem_2 = max(0, ev_monthly[month] - credit_nem_2)
        ev_cost_nem_2.append(ev_cost_under_nem_2)

        # Total cost for household + EV under NEM2
        # We start with "household_cost_no_solar" as if no solar, then subtract credit, then add EV cost
        total_cost_nem_2.append(household_cost_no_solar - credit_nem_2 + ev_cost_under_nem_2)

        # 3) Solar + NEM 3.0 + Battery
        #    - Very simplified monthly approach to battery usage
        #    - Attempt to use solar in real-time if "Daytime (Peak)" selected
        #    - Then charge battery with leftover solar
        #    - Then discharge battery for EV demand
        #    - Remainder from grid
        excess_solar_nem3 = max(0, solar_monthly[month] - household_monthly[month])
        ev_shortfall = ev_monthly[month]

        # If daytime charging, use direct solar first
        if time_of_charging == "Daytime (Peak)" and excess_solar_nem3 > 0:
            direct_solar_used = min(ev_shortfall, excess_solar_nem3)
            ev_shortfall -= direct_solar_used
            excess_solar_nem3 -= direct_solar_used

        # Charge battery with leftover solar
        if excess_solar_nem3 > 0 and battery_state < battery_capacity:
            charge_possible = min(excess_solar_nem3, battery_capacity - battery_state)
            # consider battery efficiency (roughly)
            battery_state += charge_possible * DEFAULT_BATTERY_EFFICIENCY
            excess_solar_nem3 -= charge_possible

        # Discharge battery to meet EV demand
        if ev_shortfall > 0 and battery_state > 0:
            discharge = min(ev_shortfall, battery_state)
            ev_shortfall -= discharge
            battery_state -= discharge

        # Any remaining shortfall from grid
        if time_of_charging == "Night (Super Off-Peak)":
            nem_3_cost = ev_shortfall * rates["super_off_peak"]
        else:
            nem_3_cost = ev_shortfall * rates["on_peak"]

        nem_3_battery_costs.append(nem_3_cost)
        # Household cost with no solar offset logic (just off_peak) + EV cost
        total_cost_nem_3.append(household_cost_no_solar + nem_3_cost)

    return (
        ev_cost_no_solar,
        ev_cost_nem_2,
        nem_3_battery_costs,
        total_cost_no_solar,
        total_cost_nem_2,
        total_cost_nem_3
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~ HOURLY MODEL FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~
def classify_tou_period(hour):
    """Return on_peak, off_peak, or super_off_peak based on hour of day."""
    if hour in HOUR_TOU_SCHEDULE["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour(
    hour_idx,
    solar_gen,
    house_demand,
    ev_demand,
    battery_state,
    battery_capacity
):
    """
    Simulate a single hour:
      - Subtract solar to meet demand
      - Charge battery with leftover solar
      - Discharge battery if demand remains
      - Buy remainder from grid
      - Calculate cost based on TOU
    """
    hour_of_day = hour_idx % 24
    period = classify_tou_period(hour_of_day)
    rate = HOUR_TOU_RATES[period]

    total_demand = house_demand + ev_demand

    # Use solar to offset demand
    if solar_gen >= total_demand:
        leftover_solar = solar_gen - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_gen

    # Charge battery with leftover solar (if capacity available)
    if leftover_solar > 0 and battery_state < battery_capacity:
        available_capacity = battery_capacity - battery_state
        # Need to account for charge efficiency
        solar_to_batt = min(leftover_solar, available_capacity / BATTERY_HOURLY_EFFICIENCY)
        battery_state += solar_to_batt * BATTERY_HOURLY_EFFICIENCY
        leftover_solar -= solar_to_batt

    # Discharge battery if demand remains
    if total_demand > 0 and battery_state > 0:
        discharge = min(total_demand, battery_state)
        total_demand -= discharge
        battery_state -= discharge

    # Remaining demand from grid
    grid_energy = total_demand
    cost = grid_energy * rate

    return battery_state, grid_energy, cost

def run_hourly_simulation(
    daily_household_kwh,
    daily_solar_kwh,
    daily_ev_kwh,
    battery_capacity_kwh=10.0
):
    """
    Hourly simulation for 365 days:
      - Distribute daily load/production into hourly shape
      - For each hour, apply solar, battery usage, grid purchase, and cost.
    Returns total_cost, total_grid_energy
    """
    # Example shapes (normalized to sum=1)
    # Adjust as desired or replace with real data
    household_shape = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02
    ])
    household_shape /= household_shape.sum()

    # Simple solar shape: peaks around midday
    solar_shape = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10,
        0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()
    else:
        solar_shape = np.zeros(24)

    # EV shape (assume mostly overnight charging for demonstration)
    ev_shape = np.array([
        0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if ev_shape.sum() > 0:
        ev_shape /= ev_shape.sum()
    else:
        ev_shape = np.zeros(24)

    total_cost = 0.0
    total_grid_energy = 0.0
    battery_state = 0.0

    # Sim 365 days
    for day in range(365):
        house_day = daily_household_kwh[day]
        solar_day = daily_solar_kwh[day]
        ev_day = daily_ev_kwh[day]

        # Hourly distribution
        hourly_house = household_shape * house_day
        hourly_solar = solar_shape * solar_day
        hourly_ev = ev_shape * ev_day

        for hour in range(24):
            hour_idx = day * 24 + hour
            bh, bs, be = hourly_house[hour], hourly_solar[hour], hourly_ev[hour]

            battery_state, grid_kwh, cost = simulate_hour(
                hour_idx,
                bs,
                bh,
                be,
                battery_state,
                battery_capacity_kwh
            )
            total_cost += cost
            total_grid_energy += grid_kwh

        # (Optionally reset battery daily if desired)
        # battery_state = 0.0

    return total_cost, total_grid_energy

# --------------------------------------------------------------------------------
#                                   STREAMLIT APP
# --------------------------------------------------------------------------------
def main():
    st.title("Merged Energy Simulation: Monthly vs. Hourly Approaches")

    st.sidebar.header("Simulation Parameters")
    simulation_mode = st.sidebar.selectbox("Choose Simulation Granularity", ["Monthly", "Hourly"])

    # ===== COMMON INPUTS =====
    st.sidebar.subheader("EV Parameters")
    commute_miles = st.sidebar.slider("Daily Commute Distance (miles)", 10, 100, int(DEFAULT_COMMUTE_MILES))
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days_option = st.sidebar.radio("Charging Frequency", ["Daily", "Weekdays Only"])
    days_per_week = 5 if charging_days_option == "Weekdays Only" else 7
    time_of_charging = st.sidebar.radio("Time of Charging", ["Night (Super Off-Peak)", "Daytime (Peak)"], index=0)

    # Household consumption
    st.sidebar.subheader("Household Consumption")
    household_consumption = st.sidebar.slider("Avg Daily Consumption (kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluctuation = st.sidebar.slider("Consumption Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION * 100)) / 100

    # Solar & Battery
    st.sidebar.subheader("Solar & Battery")
    solar_size = st.sidebar.slider("Solar System Size (kW)", 3, 15, int(DEFAULT_SOLAR_SIZE))
    battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 0, 20, int(DEFAULT_BATTERY_CAPACITY))

    # ==============
    # MONTHLY MODE
    # ==============
    if simulation_mode == "Monthly":
        st.header("Monthly Net Calculation")

        # 1) EV Demand
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)

        # 2) Household
        household_yearly = household_consumption * (1 + fluctuation) * 365
        household_monthly = calculate_monthly_values(household_consumption * (1 + fluctuation))

        # 3) Solar Production
        solar_yearly, solar_monthly = calculate_solar_production(solar_size)

        # 4) Calculate Costs
        (
            ev_cost_no_solar,
            ev_cost_nem_2,
            nem_3_battery_costs,
            total_cost_no_solar,
            total_cost_nem_2,
            total_cost_nem_3
        ) = calculate_monthly_costs(ev_monthly, solar_monthly, household_monthly,
                                    battery_capacity, time_of_charging)

        # ~~~~~ Build DataFrame ~~~~~
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_data = pd.DataFrame({
            "Month": months,
            "EV Consumption (kWh)": ev_monthly,
            "Household Consumption (kWh)": household_monthly,
            "Solar Production (kWh)": solar_monthly,
            "EV Cost (No Solar, $)": ev_cost_no_solar,
            "EV Cost (NEM 2.0, $)": ev_cost_nem_2,
            "EV Cost (NEM 3.0 + Batt, $)": nem_3_battery_costs,
            "Total (No Solar, $)": total_cost_no_solar,
            "Total (NEM 2.0, $)": total_cost_nem_2,
            "Total (NEM 3.0 + Batt, $)": total_cost_nem_3
        })
        st.write("### Monthly Results Table")
        st.table(monthly_data)

        # Summaries
        st.write("### Annual Summaries")
        st.write(f"**Annual EV Consumption:** {sum(ev_monthly):.1f} kWh")
        st.write(f"**Annual Household Consumption:** {sum(household_monthly):.1f} kWh")
        st.write(f"**Annual Solar Production:** {sum(solar_monthly):.1f} kWh")

        st.write(f"**Total Cost (No Solar):** ${sum(total_cost_no_solar):.2f}")
        st.write(f"**Total Cost (Solar + NEM 2.0):** ${sum(total_cost_nem_2):.2f}")
        st.write(f"**Total Cost (Solar + NEM 3.0 + Battery):** ${sum(total_cost_nem_3):.2f}")

        # ~~~~~ Some Plots ~~~~~
        st.write("### Visualizations")
        # 1) Stacked consumption
        fig1, ax1 = plt.subplots()
        ax1.bar(months, monthly_data["EV Consumption (kWh)"], label="EV")
        ax1.bar(months, monthly_data["Household Consumption (kWh)"],
                bottom=monthly_data["EV Consumption (kWh)"], label="Household")
        ax1.set_ylabel("kWh")
        ax1.set_title("Monthly Energy Consumption (EV + Household)")
        ax1.legend()
        st.pyplot(fig1)

        # 2) Solar production
        fig2, ax2 = plt.subplots()
        ax2.plot(months, monthly_data["Solar Production (kWh)"], marker="o", color="gold", label="Solar Production")
        ax2.set_ylabel("kWh")
        ax2.set_title("Monthly Solar Production")
        ax2.legend()
        st.pyplot(fig2)

        # 3) EV Cost comparison
        fig3, ax3 = plt.subplots()
        ax3.plot(months, monthly_data["EV Cost (No Solar, $)"], label="No Solar")
        ax3.plot(months, monthly_data["EV Cost (NEM 2.0, $)"], label="NEM 2.0")
        ax3.plot(months, monthly_data["EV Cost (NEM 3.0 + Batt, $)"], label="NEM 3.0 + Battery")
        ax3.set_ylabel("Cost ($)")
        ax3.set_title("Monthly EV Charging Costs")
        ax3.legend()
        st.pyplot(fig3)

        # 4) Total cost comparison
        fig4, ax4 = plt.subplots()
        ax4.plot(months, monthly_data["Total (No Solar, $)"], label="No Solar")
        ax4.plot(months, monthly_data["Total (NEM 2.0, $)"], label="Solar + NEM 2.0")
        ax4.plot(months, monthly_data["Total (NEM 3.0 + Batt, $)"], label="Solar + Batt (NEM 3.0)")
        ax4.set_ylabel("Cost ($)")
        ax4.set_title("Monthly Total Costs")
        ax4.legend()
        st.pyplot(fig4)

    # ==============
    # HOURLY MODE
    # ==============
    else:
        st.header("Hourly Time-Step Simulation")

        # --- Build daily arrays (365 days) for household, solar, EV ---
        # 1) EV demand (kWh/day)
        #    daily_miles = commute_miles every day or for days_per_week?
        #    We'll do a simple approach: average it out across the full year
        daily_ev_kwh = (commute_miles / efficiency) * (days_per_week / 7.0)

        daily_household = household_consumption * (1 + fluctuation)  # average day
        # Very rough solar estimate: 4 kWh/kW/day
        daily_solar = solar_size * 4

        # Make 365-day arrays
        daily_household_kwh = np.full(365, daily_household)
        daily_solar_kwh = np.full(365, daily_solar)
        daily_ev_kwh = np.full(365, daily_ev_kwh)

        # --- Run 3 scenarios for comparison ---
        # 1) No Solar (battery doesn't matter)
        cost_no_solar, grid_no_solar = run_hourly_simulation(
            daily_household_kwh,
            np.zeros(365),  # no solar
            daily_ev_kwh,
            battery_capacity_kwh=0
        )

        # 2) Solar Only (no battery)
        cost_solar_nobatt, grid_solar_nobatt = run_hourly_simulation(
            daily_household_kwh,
            daily_solar_kwh,
            daily_ev_kwh,
            battery_capacity_kwh=0
        )

        # 3) Solar + Battery
        cost_solar_batt, grid_solar_batt = run_hourly_simulation(
            daily_household_kwh,
            daily_solar_kwh,
            daily_ev_kwh,
            battery_capacity_kwh=battery_capacity
        )

        # --- Display results ---
        st.write("### Annual Results (Hourly Simulation)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("No Solar Cost ($/yr)", f"{cost_no_solar:,.2f}")
            st.metric("Grid kWh (No Solar)", f"{grid_no_solar:,.0f}")
        with col2:
            st.metric("Solar Only Cost ($/yr)", f"{cost_solar_nobatt:,.2f}")
            st.metric("Grid kWh (Solar)", f"{grid_solar_nobatt:,.0f}")
        with col3:
            st.metric("Solar+Battery Cost ($/yr)", f"{cost_solar_batt:,.2f}")
            st.metric("Grid kWh (Solar+Batt)", f"{grid_solar_batt:,.0f}")

        st.write("""
        **Notes on Hourly Approach**  
        - A simple load/solar shape is assumed every day (no seasonality).  
        - EV is assumed to charge mostly at night.  
        - Battery usage is naive (use battery whenever there's demand).  
        - TOU rates are applied per hour based on static on/off/super-off-peak periods.  
        - In reality, you'll want to handle net export compensation if you're modeling NEM 3.0 in detail.
        """)

if __name__ == "__main__":
    main()
