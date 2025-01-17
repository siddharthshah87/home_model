import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#                               GLOBAL CONSTANTS
# --------------------------------------------------------------------------------

# --- Monthly Model Constants ---
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
SUMMER_MONTHS = [5, 6, 7, 8]  # June(5) to Sept(8)
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]  # Jan-May, Oct-Dec

# --- Hourly Model Constants ---
HOUR_TOU_SCHEDULE = {
    "on_peak_hours": list(range(16, 21)),       # 4 PM - 8 PM
    "off_peak_hours": list(range(7, 16)) + [21, 22],  # 7 AM - 3 PM, 9 PM - 10 PM
    "super_off_peak_hours": list(range(0, 7)) + [23],  # 0 AM - 6 AM, 11 PM
}
HOUR_TOU_RATES = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY = 0.90
DAYS_PER_YEAR = 365

# --------------------------------------------------------------------------------
#                            MONTHLY MODEL FUNCTIONS
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    """Multiply a daily_value by the # of days in each month to get monthly values."""
    return [daily_value * days for days in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    """
    Returns (yearly_kWh, monthly_kWh_list).
    - miles: daily commute
    - efficiency: miles/kWh
    - days_per_week: 5 or 7 (weekdays or daily)
    """
    daily_demand = miles / efficiency
    total_days = days_per_week * 52  # ~364 days
    yearly_demand = daily_demand * total_days
    monthly_demand = calculate_monthly_values(daily_demand * (days_per_week / 7.0))
    return yearly_demand, monthly_demand

def calculate_solar_production(size_kw):
    """
    Returns (yearly_kWh, monthly_kWh_list).
    Very rough assumption: 4 kWh/kW/day
    """
    yearly = size_kw * 4 * 365
    monthly = calculate_monthly_values(size_kw * 4)
    return yearly, monthly

def calculate_monthly_costs(ev_monthly, solar_monthly, household_monthly,
                            battery_capacity, time_of_charging):
    """
    Monthly net approach, comparing:
      1) No solar
      2) Solar + NEM 2.0
      3) Solar + NEM 3.0 + battery (naive monthly logic)
    """
    ev_cost_no_solar = []
    ev_cost_nem_2 = []
    ev_cost_nem_3 = []

    total_no_solar = []
    total_nem_2 = []
    total_nem_3 = []

    battery_state = 0.0  # naive approach: carry over month to month

    for month in range(12):
        # Determine rates by season
        if month in SUMMER_MONTHS:
            rates = TOU_RATES["summer"]
        else:
            rates = TOU_RATES["winter"]

        # 1) No Solar scenario
        #    - Household at off-peak
        #    - EV at super-off-peak
        household_cost_ns = household_monthly[month] * rates["off_peak"]
        ev_ns = ev_monthly[month] * rates["super_off_peak"]
        ev_cost_no_solar.append(ev_ns)
        total_no_solar.append(household_cost_ns + ev_ns)

        # 2) Solar + NEM 2.0
        #    - Excess solar credited at off_peak
        #    - EV usage net against that credit
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit_nem_2 = max(0, excess_solar * rates["off_peak"])  # simplistic
        ev_nem_2 = max(0, ev_monthly[month] - credit_nem_2)
        ev_cost_nem_2.append(ev_nem_2)

        # total cost under NEM 2.0
        total_nem_2.append(household_cost_ns - credit_nem_2 + ev_nem_2)

        # 3) Solar + NEM 3.0 + battery (extremely naive monthly approach)
        #    - Attempt "daytime" usage from solar if time_of_charging == "Daytime (Peak)"
        #    - Then charge battery with leftover solar
        #    - Then discharge battery for EV if needed
        #    - Remainder from grid
        # We'll reuse household_cost_ns as baseline for household at off-peak
        # (No attempt to net solar against household for cost offsets)
        excess_solar_nem3 = max(0, solar_monthly[month] - household_monthly[month])
        ev_shortfall = ev_monthly[month]

        if time_of_charging == "Daytime (Peak)" and excess_solar_nem3 > 0:
            direct_solar = min(ev_shortfall, excess_solar_nem3)
            ev_shortfall -= direct_solar
            excess_solar_nem3 -= direct_solar

        # Charge battery
        if excess_solar_nem3 > 0 and battery_state < battery_capacity:
            can_charge = min(excess_solar_nem3, battery_capacity - battery_state)
            battery_state += can_charge * DEFAULT_BATTERY_EFFICIENCY
            excess_solar_nem3 -= can_charge

        # Discharge battery
        if ev_shortfall > 0 and battery_state > 0:
            discharge = min(ev_shortfall, battery_state)
            ev_shortfall -= discharge
            battery_state -= discharge

        # Remainder from grid
        if time_of_charging == "Night (Super Off-Peak)":
            ev_cost = ev_shortfall * rates["super_off_peak"]
        else:
            ev_cost = ev_shortfall * rates["on_peak"]

        ev_cost_nem_3.append(ev_cost)
        total_nem_3.append(household_cost_ns + ev_cost)

    return (
        ev_cost_no_solar,
        ev_cost_nem_2,
        ev_cost_nem_3,
        total_no_solar,
        total_nem_2,
        total_nem_3
    )

# --------------------------------------------------------------------------------
#                            HOURLY MODEL FUNCTIONS
# --------------------------------------------------------------------------------

def classify_tou_period(hour_of_day):
    if hour_of_day in HOUR_TOU_SCHEDULE["on_peak_hours"]:
        return "on_peak"
    elif hour_of_day in HOUR_TOU_SCHEDULE["off_peak_hours"]:
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
    Naive approach:
      - Use solar first
      - leftover solar charges battery
      - battery discharges if demand remains
      - remainder from grid
      - no net export credit
    """
    hour_of_day = hour_idx % 24
    period = classify_tou_period(hour_of_day)
    rate = HOUR_TOU_RATES[period]

    total_demand = house_kwh + ev_kwh
    if solar_kwh >= total_demand:
        leftover_solar = solar_kwh - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_kwh

    # Charge battery with leftover solar
    solar_unused = 0
    if leftover_solar > 0 and battery_state < battery_capacity:
        can_store = (battery_capacity - battery_state) / BATTERY_HOURLY_EFFICIENCY
        solar_to_battery = min(leftover_solar, can_store)
        battery_state += solar_to_battery * BATTERY_HOURLY_EFFICIENCY
        leftover_solar -= solar_to_battery
        solar_unused = leftover_solar
    else:
        solar_unused = leftover_solar

    # Discharge battery if demand remains
    if total_demand > 0 and battery_state > 0:
        discharge = min(total_demand, battery_state)
        total_demand -= discharge
        battery_state -= discharge

    # Remainder from grid
    grid_kwh = total_demand
    cost = grid_kwh * rate

    return battery_state, grid_kwh, cost, solar_unused

def run_hourly_simulation(
    daily_household,
    daily_solar,
    daily_ev,
    battery_capacity_kwh=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
    """
    Hourly sim for 365 days with naive approach. 
    No net export credit; leftover solar is "lost".
    Returns (total_cost, total_grid, total_solar_unused, df_hourly).
    """
    # Shapes (24-hour) repeated daily
    household_shape = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02
    ])
    household_shape /= household_shape.sum()

    solar_shape = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10,
        0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()

    if ev_charging_pattern == "Night":
        # Heavy charging from midnight to ~5 AM
        ev_shape = np.array([
            0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    else:
        # Example "Daytime" pattern for EV
        ev_shape = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15,
            0.15, 0.15, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    if ev_shape.sum() > 0:
        ev_shape /= ev_shape.sum()

    total_cost = 0.0
    total_grid = 0.0
    total_solar_unused = 0.0

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

    battery_state = 0.0

    for d in range(DAYS_PER_YEAR):
        # Possibly reset battery each morning
        if reset_battery_daily:
            battery_state = 0.0

        # daily distribution
        dh = daily_household[d]
        ds = daily_solar[d]
        de = daily_ev[d]

        hh_24 = dh * household_shape
        ss_24 = ds * solar_shape
        ee_24 = de * ev_shape

        for hour in range(24):
            hour_idx = d*24 + hour
            h_h = hh_24[hour]
            h_s = ss_24[hour]
            h_e = ee_24[hour]

            battery_state, grid_kwh, cost, solar_unused = simulate_hour(
                hour_idx,
                h_s,
                h_h,
                h_e,
                battery_state,
                battery_capacity_kwh
            )

            total_cost += cost
            total_grid += grid_kwh
            total_solar_unused += solar_unused

            # log results
            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(h_h)
            results["ev_kwh"].append(h_e)
            results["solar_kwh"].append(h_s)
            results["grid_kwh"].append(grid_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(solar_unused)

    df_hourly = pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df_hourly

# --------------------------------------------------------------------------------
#                                STREAMLIT APP
# --------------------------------------------------------------------------------

def main():
    st.title("Unified App: Monthly & Hourly Energy Simulation")

    st.sidebar.header("Common User Inputs")

    # EV inputs
    commute_miles = st.sidebar.slider("Daily Commute Distance (miles)", 10, 100, DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days_option = st.sidebar.radio("Charging Frequency", ["Daily", "Weekdays Only"])
    days_per_week = 5 if charging_days_option == "Weekdays Only" else 7
    time_of_charging = st.sidebar.radio(
        "Time of Charging (Monthly model)",
        ["Night (Super Off-Peak)", "Daytime (Peak)"],
        index=0
    )

    # Household
    household_consumption = st.sidebar.slider("Avg Daily Household (kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluctuation = st.sidebar.slider("Consumption Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION*100)) / 100

    # Solar & Battery
    solar_size = st.sidebar.slider("Solar System Size (kW)", 3, 15, int(DEFAULT_SOLAR_SIZE))
    battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 0, 20, int(DEFAULT_BATTERY_CAPACITY))

    # We can have a separate toggle for Hourly EV pattern
    ev_charging_hourly = st.sidebar.selectbox("EV Charging Pattern (Hourly model)", ["Night", "Daytime"])
    reset_battery_daily = st.sidebar.checkbox("Reset Battery Each Day? (Hourly model)", False)

    # Create Tabs
    tab1, tab2 = st.tabs(["Monthly Approach", "Hourly Approach"])

    # --------------------------------------------------------------------------------
    #                            TAB 1: MONTHLY
    # --------------------------------------------------------------------------------
    with tab1:
        st.header("Monthly Net Calculation")

        # 1) EV Demand
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)

        # 2) Household
        daily_house = household_consumption * (1 + fluctuation)
        household_yearly = daily_house * 365
        household_monthly = calculate_monthly_values(daily_house)

        # 3) Solar
        solar_yearly, solar_monthly = calculate_solar_production(solar_size)

        # 4) Costs
        (
            ev_cost_no_solar,
            ev_cost_nem_2,
            ev_cost_nem_3,
            total_cost_no_solar,
            total_cost_nem_2,
            total_cost_nem_3
        ) = calculate_monthly_costs(
            ev_monthly,
            solar_monthly,
            household_monthly,
            battery_capacity,
            time_of_charging
        )

        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_data = pd.DataFrame({
            "Month": months,
            "EV (kWh)": ev_monthly,
            "House (kWh)": household_monthly,
            "Solar (kWh)": solar_monthly,
            "EV Cost (No Solar)": ev_cost_no_solar,
            "EV Cost (NEM 2)": ev_cost_nem_2,
            "EV Cost (NEM 3 + Batt)": ev_cost_nem_3,
            "Total (No Solar)": total_cost_no_solar,
            "Total (NEM 2)": total_cost_nem_2,
            "Total (NEM 3 + Batt)": total_cost_nem_3
        })

        st.subheader("Monthly Results Table")
        st.dataframe(monthly_data.style.format(precision=2))

        st.write("#### Annual Summaries")
        st.write(f"**Annual EV Consumption**: {sum(ev_monthly):.1f} kWh")
        st.write(f"**Annual Household Consumption**: {sum(household_monthly):.1f} kWh")
        st.write(f"**Annual Solar Production**: {sum(solar_monthly):.1f} kWh")

        st.write(f"**Total Cost (No Solar)**: ${sum(total_cost_no_solar):.2f}")
        st.write(f"**Total Cost (Solar + NEM 2)**: ${sum(total_cost_nem_2):.2f}")
        st.write(f"**Total Cost (Solar + NEM 3 + Batt)**: ${sum(total_cost_nem_3):.2f}")

        # Plots
        st.write("### Monthly Energy Consumption")
        fig1, ax1 = plt.subplots()
        ax1.bar(months, monthly_data["EV (kWh)"], label="EV")
        ax1.bar(months, monthly_data["House (kWh)"], bottom=monthly_data["EV (kWh)"], label="House")
        ax1.set_ylabel("kWh")
        ax1.legend()
        st.pyplot(fig1)

        st.write("### Solar Production")
        fig2, ax2 = plt.subplots()
        ax2.plot(months, monthly_data["Solar (kWh)"], color="gold", marker="o", label="Solar")
        ax2.set_ylabel("kWh")
        ax2.legend()
        st.pyplot(fig2)

        st.write("### EV Charging Cost Comparison")
        fig3, ax3 = plt.subplots()
        ax3.plot(months, monthly_data["EV Cost (No Solar)"], label="No Solar")
        ax3.plot(months, monthly_data["EV Cost (NEM 2)"], label="NEM 2")
        ax3.plot(months, monthly_data["EV Cost (NEM 3 + Batt)"], label="NEM 3 + Batt")
        ax3.set_ylabel("Cost ($)")
        ax3.legend()
        st.pyplot(fig3)

        st.write("### Total Monthly Cost Comparison")
        fig4, ax4 = plt.subplots()
        ax4.plot(months, monthly_data["Total (No Solar)"], label="No Solar")
        ax4.plot(months, monthly_data["Total (NEM 2)"], label="NEM 2")
        ax4.plot(months, monthly_data["Total (NEM 3 + Batt)"], label="NEM 3 + Batt")
        ax4.set_ylabel("Cost ($)")
        ax4.legend()
        st.pyplot(fig4)

        st.write("""
        **Notes (Monthly Approach)**  
        - Simplified net metering assumptions (credits at off-peak rate, etc.).  
        - Battery usage is aggregated monthly (not daily).  
        - Real NEM 3.0 typically involves **hourly** export rates, different credits, etc.  
        """)

    # --------------------------------------------------------------------------------
    #                            TAB 2: HOURLY
    # --------------------------------------------------------------------------------
    with tab2:
        st.header("Hourly Time-Step Approach")

        # Build 365-day arrays for house, solar, ev
        # Rough daily solar production: 4 kWh/kW/day
        daily_solar_kwh = np.full(DAYS_PER_YEAR, solar_size * 4)
        # Household: (household_consumption * (1+fluctuation)) daily
        daily_house_kwh = np.full(DAYS_PER_YEAR, household_consumption * (1 + fluctuation))
        # EV: daily miles / efficiency, scaled by how many days/week
        daily_ev_kwh = np.full(DAYS_PER_YEAR, (commute_miles / efficiency) * (days_per_week / 7.0))

        total_cost, total_grid, total_solar_unused, df_hourly = run_hourly_simulation(
            daily_house_kwh,
            daily_solar_kwh,
            daily_ev_kwh,
            battery_capacity_kwh=battery_capacity,
            ev_charging_pattern=ev_charging_hourly,
            reset_battery_daily=reset_battery_daily
        )

        st.write("### Annual Results")
        st.write(f"**Total Annual Cost:** ${total_cost:,.2f}")
        st.write(f"**Total Grid Usage:** {total_grid:,.0f} kWh")
        st.write(f"**Unused Solar (lost/no export):** {total_solar_unused:,.0f} kWh")

        st.write(f"**Average Daily Cost:** ${total_cost / DAYS_PER_YEAR:,.2f}")
        st.write(f"**Average Daily Grid:** {total_grid / DAYS_PER_YEAR:,.1f} kWh")
        st.write(f"**Average Daily Unused Solar:** {total_solar_unused / DAYS_PER_YEAR:,.1f} kWh")

        st.write("""
        **Notes (Hourly Approach)**  
        - A simple load/solar shape is repeated every day (no seasonality).  
        - EV charging pattern set to either 'Night' or 'Daytime' (static profile).  
        - Battery usage is naive: charge whenever there's leftover solar, discharge whenever there's demand.  
        - **No net export compensation**: leftover solar is lost.  
        - TOU rates are fixed daily all year (on-peak/off-peak/super-off-peak).  
        """)

        # Show day selection for plot
        day_to_plot = st.slider("Select Day to Plot (0-364)", 0, DAYS_PER_YEAR-1, 0)
        df_day = df_hourly[df_hourly["day"] == day_to_plot]

        st.write(f"### Hourly Profile for Day {day_to_plot}")
        figA, axA = plt.subplots()
        axA.plot(df_day["hour"], df_day["house_kwh"], label="House Load")
        axA.plot(df_day["hour"], df_day["ev_kwh"], label="EV Load", linestyle="--")
        axA.plot(df_day["hour"], df_day["solar_kwh"], label="Solar", color="gold")
        axA.set_xlabel("Hour of Day")
        axA.set_ylabel("kWh")
        axA.legend()
        axA.set_title(f"Day {day_to_plot} Hourly Profile")
        st.pyplot(figA)

        st.write("#### Battery State Over That Day")
        figB, axB = plt.subplots()
        axB.plot(df_day["hour"], df_day["battery_state"], label="Battery State (kWh)", color="green")
        axB.set_xlabel("Hour of Day")
        axB.set_ylabel("kWh in Battery")
        axB.legend()
        axB.set_title(f"Day {day_to_plot} - Battery State")
        st.pyplot(figB)

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
