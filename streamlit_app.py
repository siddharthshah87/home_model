import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#                               CONSTANTS & SETUP
# --------------------------------------------------------------------------------

# Time-of-Use schedule (static year-round, simplified)
# Hours of the day (0-23) grouped into three categories:
TOU_SCHEDULE = {
    "on_peak_hours": list(range(16, 21)),  # 4 PM - 8 PM
    "off_peak_hours": list(range(7, 16)) + [21, 22],  # 7 AM - 3 PM, 9 PM-10 PM
    "super_off_peak_hours": list(range(0, 7)) + [23],  # 0 AM - 6 AM, 11 PM
}

TOU_RATES = {
    "on_peak": 0.45,          # $/kWh
    "off_peak": 0.25,         # $/kWh
    "super_off_peak": 0.12,   # $/kWh
}

# Battery charging/discharging efficiency
BATTERY_EFFICIENCY = 0.90

# Number of days in this simulation (default 365)
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = DAYS_PER_YEAR * 24

# --------------------------------------------------------------------------------
#                            HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def classify_tou_period(hour_of_day):
    """
    Given the hour_of_day (0-23), return one of: 'on_peak', 'off_peak', or 'super_off_peak'.
    """
    if hour_of_day in TOU_SCHEDULE["on_peak_hours"]:
        return "on_peak"
    elif hour_of_day in TOU_SCHEDULE["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour(
    hour_idx,
    solar_kwh,
    house_kwh,
    ev_kwh,
    battery_state,
    battery_capacity
):
    """
    Simulate one hour of operation under a naive strategy:
      1. Use solar to meet immediate demand.
      2. If leftover solar, charge the battery (up to capacity).
      3. If demand remains after solar, discharge the battery if possible.
      4. If demand is still unmet, buy from the grid.
    
    Returns:
      - new_battery_state
      - grid_kwh (how many kWh we pulled from the grid)
      - cost (grid cost for this hour)
      - solar_unused_kwh (leftover solar that can't be stored because battery is full)
    """
    # Determine which hour of the day (0-23)
    hour_of_day = hour_idx % 24  
    period = classify_tou_period(hour_of_day)
    rate = TOU_RATES[period]

    total_demand = house_kwh + ev_kwh

    # Step 1: Use solar to meet immediate demand
    if solar_kwh >= total_demand:
        leftover_solar = solar_kwh - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_kwh

    # Step 2: Charge battery with leftover solar, if battery has capacity
    solar_unused_kwh = 0
    if leftover_solar > 0 and battery_state < battery_capacity:
        # how many kWh can we store?
        capacity_needed = battery_capacity - battery_state
        # Because of charge efficiency, you need more kWh input to store 1 kWh
        # but let's simplify: leftover_solar is input, battery gets leftover_solar * EFFICIENCY
        # We'll see how many kWh we can input from leftover_solar
        max_solar_input = capacity_needed / BATTERY_EFFICIENCY

        # actual solar we can put in the battery
        solar_to_battery = min(leftover_solar, max_solar_input)
        
        # battery gains...
        battery_state += solar_to_battery * BATTERY_EFFICIENCY
        # leftover_solar used up
        leftover_solar -= solar_to_battery

        # If there's still leftover_solar after filling battery, it's "unused" 
        # (i.e., exported or lost if we are not modeling NEM export).
        solar_unused_kwh = leftover_solar
    else:
        # No battery capacity or no leftover solar -> leftover_solar is simply unused
        solar_unused_kwh = leftover_solar

    # Step 3: Discharge battery if demand remains
    if total_demand > 0 and battery_state > 0:
        discharge_amount = min(total_demand, battery_state)
        total_demand -= discharge_amount
        battery_state -= discharge_amount

    # Step 4: Any remaining demand is met by the grid
    grid_kwh = total_demand
    cost = grid_kwh * rate

    return battery_state, grid_kwh, cost, solar_unused_kwh

def run_hourly_simulation(
    daily_household,
    daily_solar,
    daily_ev,
    battery_capacity_kwh=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
    """
    Run a naive hourly simulation for a year (365 days), under these assumptions:
      - daily_household: array (365) with daily household consumption (kWh/day)
      - daily_solar:     array (365) with daily solar production (kWh/day)
      - daily_ev:        array (365) with daily EV consumption (kWh/day)
      - battery_capacity_kwh: size of battery in kWh
      - ev_charging_pattern: if 'Night', we assume a certain shape for EV usage
      - reset_battery_daily: if True, battery_state = 0 at the start of each day

    Returns total results over the year and an hourly DataFrame log:
      (total_cost, total_grid_kwh, total_solar_unused, df_hourly)
    """

    # 1) Build 24-hour SHAPES for household, solar, and EV
    # Simplified shapes that sum to 1, repeated for each day
    household_shape = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02
    ])
    household_shape /= household_shape.sum()

    # Peak solar midday
    solar_shape = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10,
        0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()

    # EV shape for "Night" charging (heavy usage from ~0-5 AM)
    if ev_charging_pattern == "Night":
        ev_shape = np.array([
            0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    else:
        # Example "Daytime" shape (charging midday)
        ev_shape = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15,
            0.15, 0.15, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    if ev_shape.sum() > 0:
        ev_shape /= ev_shape.sum()

    # 2) Initialize totals and logging
    total_cost = 0.0
    total_grid_kwh = 0.0
    total_solar_unused = 0.0
    battery_state = 0.0

    # Optional: store hourly data for analysis
    results = {
        "day": [],
        "hour": [],
        "house_kwh": [],
        "ev_kwh": [],
        "solar_kwh": [],
        "grid_kwh": [],
        "cost": [],
        "battery_state": [],
        "solar_unused": [],
    }

    # 3) Run simulation day by day
    for day in range(DAYS_PER_YEAR):
        # daily totals
        dh = daily_household[day]
        ds = daily_solar[day]
        de = daily_ev[day]

        # Distribute over 24 hours
        house_24 = dh * household_shape
        solar_24 = ds * solar_shape
        ev_24 = de * ev_shape

        # If resetting battery each day:
        if reset_battery_daily:
            battery_state = 0.0

        for hour in range(24):
            hour_idx = day*24 + hour
            h_h = house_24[hour]
            h_s = solar_24[hour]
            h_e = ev_24[hour]

            battery_state, grid_kwh, cost, solar_unused = simulate_hour(
                hour_idx,
                h_s,
                h_h,
                h_e,
                battery_state,
                battery_capacity_kwh
            )
            
            total_cost += cost
            total_grid_kwh += grid_kwh
            total_solar_unused += solar_unused

            # log
            results["day"].append(day)
            results["hour"].append(hour)
            results["house_kwh"].append(h_h)
            results["ev_kwh"].append(h_e)
            results["solar_kwh"].append(h_s)
            results["grid_kwh"].append(grid_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(solar_unused)

    # Build DataFrame
    df_hourly = pd.DataFrame(results)
    return total_cost, total_grid_kwh, total_solar_unused, df_hourly

# --------------------------------------------------------------------------------
#                         STREAMLIT APP DEMO
# --------------------------------------------------------------------------------
def main():
    st.title("Hourly Time-Step Simulation (Simple Approach)")

    st.write("""
    This example demonstrates an **hourly simulation** with:
    - A simple daily shape for **household load** and **solar production** used every day (no seasonality).
    - An **EV charging profile** that is mostly **night** (or optionally midday).
    - A **naive battery dispatch** (use battery whenever there's demand, charge whenever solar is available).
    - **Static TOU rates** year-round (on-peak/off-peak/super-off-peak) without net export compensation.
    """)

    # ---- SIDEBAR INPUTS ----
    st.sidebar.header("Simulation Inputs")

    # 1) Household
    avg_house_kwh = st.sidebar.slider("Household Daily kWh", 10, 60, 20)
    # We'll keep it the same for all 365 days in this demo

    # 2) Solar
    solar_kw = st.sidebar.slider("Solar System (kW)", 3, 15, 7)
    # Rough daily production = 4 kWh/kW/day
    solar_daily_kwh = solar_kw * 4

    # 3) EV
    daily_miles = st.sidebar.slider("Daily Miles Driven", 10, 100, 30)
    ev_efficiency = st.sidebar.slider("EV Efficiency (miles/kWh)", 3.0, 5.0, 4.0)
    ev_charging_pattern = st.sidebar.selectbox("EV Charging Pattern", ["Night", "Daytime"])
    # We'll assume user charges daily
    ev_daily_kwh = daily_miles / ev_efficiency

    # 4) Battery
    battery_capacity_kwh = st.sidebar.slider("Battery Capacity (kWh)", 0, 20, 10)
    reset_daily = st.sidebar.checkbox("Reset Battery to 0 each day?", False)

    # ---- BUILD 365-DAY ARRAYS ----
    daily_household = np.full(DAYS_PER_YEAR, avg_house_kwh)
    daily_solar = np.full(DAYS_PER_YEAR, solar_daily_kwh)
    daily_ev = np.full(DAYS_PER_YEAR, ev_daily_kwh)

    # ---- RUN SIMULATION ----
    total_cost, total_grid, total_solar_unused, df_hourly = run_hourly_simulation(
        daily_household,
        daily_solar,
        daily_ev,
        battery_capacity_kwh=battery_capacity_kwh,
        ev_charging_pattern=ev_charging_pattern,
        reset_battery_daily=reset_daily
    )

    # ---- RESULTS ----
    st.subheader("Simulation Results (Annual)")

    st.write(f"**Total Annual Cost:** ${total_cost:,.2f}")
    st.write(f"**Total Grid Usage:** {total_grid:,.1f} kWh")
    st.write(f"**Unused Solar (lost / no export):** {total_solar_unused:,.1f} kWh")

    # Optionally, show average daily values
    avg_daily_cost = total_cost / DAYS_PER_YEAR
    avg_daily_grid = total_grid / DAYS_PER_YEAR
    avg_daily_unused = total_solar_unused / DAYS_PER_YEAR

    st.write(f"**Avg Daily Cost:** ${avg_daily_cost:.2f}")
    st.write(f"**Avg Daily Grid Usage:** {avg_daily_grid:.1f} kWh")
    st.write(f"**Avg Daily Unused Solar:** {avg_daily_unused:.1f} kWh")

    # ----- Plot Some Results -----
    st.subheader("Hourly Data Plots")
    
    # Example 1: Show a random day or a user-selected day
    day_to_plot = st.slider("Select a Day to Plot", 0, DAYS_PER_YEAR-1, 0)
    df_day = df_hourly[df_hourly["day"] == day_to_plot].copy()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_day["hour"], df_day["house_kwh"], label="House Load (kWh)")
    ax.plot(df_day["hour"], df_day["ev_kwh"], label="EV Load (kWh)", linestyle="--")
    ax.plot(df_day["hour"], df_day["solar_kwh"], label="Solar (kWh)", color="gold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("kWh")
    ax.set_title(f"Day {day_to_plot} Profiles")
    ax.legend()
    st.pyplot(fig)

    # Example 2: Battery State each hour of the selected day
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(df_day["hour"], df_day["battery_state"], label="Battery State (kWh)", color="green")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Battery State (kWh)")
    ax2.set_title(f"Day {day_to_plot} - Battery State Over Time")
    ax2.legend()
    st.pyplot(fig2)

    st.write("""
    ### Notes on Hourly Approach
    - **Simple load/solar shape** is assumed every day (no seasonality).
    - **EV is assumed to charge mostly at night** (if 'Night' selected).
    - **Battery usage is naive**: any leftover solar charges the battery immediately; any demand discharges it.
    - **TOU rates** are applied each hour, based on a fixed on/off-peak schedule (the same all year).
    - **No net export compensation** is modeled, so any solar exceeding demand + battery capacity is "lost".
    """)

if __name__ == "__main__":
    main()
