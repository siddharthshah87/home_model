import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#                          GLOBAL CONSTANTS
# --------------------------------------------------------------------------------

# ~~~~~~~~~ MONTHLY MODEL CONSTANTS ~~~~~~~~~
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
SUMMER_MONTHS = [5, 6, 7, 8]
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]

# ~~~~~~~~~ BASIC HOURLY MODEL CONSTANTS ~~~~~~~~~
HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16, 21)),        # 4 PM - 8 PM
    "off_peak_hours": list(range(7, 16)) + [21, 22],  # 7 AM - 3 PM, 9 PM - 10 PM
    "super_off_peak_hours": list(range(0, 7)) + [23], # 0 AM - 6 AM, 11 PM
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90

# ~~~~~~~~~ ADVANCED HOURLY "NEM 3.0-LIKE" CONSTANTS ~~~~~~~~~
ADV_TOU_SCHEDULE = {
    "on_peak_hours": list(range(16, 21)),       
    "off_peak_hours": list(range(7, 16)) + [21, 22],
    "super_off_peak_hours": list(range(0, 7)) + [23],
}
IMPORT_RATES = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
EXPORT_RATES = {
    "on_peak": 0.10,
    "off_peak": 0.06,
    "super_off_peak": 0.04
}
ADV_BATTERY_EFFICIENCY = 0.90

DAYS_PER_YEAR = 365

# --------------------------------------------------------------------------------
#                  COMMON / SEASONAL HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def build_daily_array_seasonal(base_daily_value, monthly_factors):
    """
    Create a 365-element array of daily values with
    per-month scaling factors.
      - base_daily_value: baseline daily usage/production
      - monthly_factors: list of length 12 for each month
    """
    daily_values = []
    for month_idx, ndays in enumerate(DAYS_IN_MONTH):
        factor = monthly_factors[month_idx]
        for _ in range(ndays):
            daily_values.append(base_daily_value * factor)
    # Trim/ensure length is exactly 365
    return np.array(daily_values[:365])

# --------------------------------------------------------------------------------
#                           MONTHLY MODEL FUNCTIONS
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    """Multiply a daily value by the # of days in each month."""
    return [daily_value * d for d in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    """
    Return (yearly_kWh, monthly_kWh_list).
    - miles: daily commute
    - efficiency: miles/kWh
    - days_per_week: e.g. 5 or 7
    """
    daily_demand = miles / efficiency
    total_days = days_per_week * 52  # ~364
    yearly = daily_demand * total_days
    monthly = calculate_monthly_values(daily_demand * (days_per_week / 7.0))
    return yearly, monthly

def calculate_solar_production(size_kw):
    """4 kWh/kW/day as a rough average."""
    yearly = size_kw * 4 * 365
    monthly = calculate_monthly_values(size_kw * 4)
    return yearly, monthly

def calculate_monthly_costs(ev_monthly, solar_monthly, household_monthly,
                            battery_capacity, time_of_charging):
    """
    Compare:
      1) No solar
      2) Solar + NEM 2.0
      3) Solar + NEM 3.0 + battery (naive monthly)
    """
    ev_cost_no_solar, ev_cost_nem_2, ev_cost_nem_3 = [], [], []
    total_no_solar, total_nem_2, total_nem_3 = [], [], []

    battery_state = 0.0

    for month in range(12):
        # Pick summer/winter rates
        if month in SUMMER_MONTHS:
            rates = TOU_RATES["summer"]
        else:
            rates = TOU_RATES["winter"]

        # 1) No Solar
        cost_house_ns = household_monthly[month] * rates["off_peak"]
        cost_ev_ns = ev_monthly[month] * rates["super_off_peak"]
        ev_cost_no_solar.append(cost_ev_ns)
        total_no_solar.append(cost_house_ns + cost_ev_ns)

        # 2) NEM 2.0
        # Excess solar credited at off_peak; net that credit from EV usage
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit_nem2 = max(0, excess_solar * rates["off_peak"])
        cost_ev_nem2 = max(0, ev_monthly[month] - credit_nem2)
        ev_cost_nem_2.append(cost_ev_nem2)
        total_nem_2.append(cost_house_ns - credit_nem2 + cost_ev_nem2)

        # 3) NEM 3.0 + battery (naive monthly approach)
        leftover_solar = max(0, solar_monthly[month] - household_monthly[month])
        ev_shortfall = ev_monthly[month]

        if time_of_charging == "Daytime (Peak)" and leftover_solar > 0:
            direct_solar = min(ev_shortfall, leftover_solar)
            ev_shortfall -= direct_solar
            leftover_solar -= direct_solar

        # Charge battery
        if leftover_solar > 0 and battery_state < battery_capacity:
            can_charge = min(leftover_solar, battery_capacity - battery_state)
            battery_state += can_charge * DEFAULT_BATTERY_EFFICIENCY
            leftover_solar -= can_charge

        # Discharge battery
        if ev_shortfall > 0 and battery_state > 0:
            discharge = min(ev_shortfall, battery_state)
            ev_shortfall -= discharge
            battery_state -= discharge

        # Remainder from grid
        if time_of_charging == "Night (Super Off-Peak)":
            cost_ev_nem3 = ev_shortfall * rates["super_off_peak"]
        else:
            cost_ev_nem3 = ev_shortfall * rates["on_peak"]

        ev_cost_nem_3.append(cost_ev_nem3)
        total_nem_3.append(cost_house_ns + cost_ev_nem3)

    return ev_cost_no_solar, ev_cost_nem_2, ev_cost_nem_3, total_no_solar, total_nem_2, total_nem_3

# --------------------------------------------------------------------------------
#                  BASIC HOURLY MODEL FUNCTIONS
# --------------------------------------------------------------------------------

def classify_tou_basic(hour_of_day):
    if hour_of_day in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour_of_day in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
    """
    Very naive approach:
    1) Use solar to meet demand
    2) leftover solar -> battery
    3) discharge battery if demand remains
    4) remainder from grid
    5) single set of TOU rates for cost
    6) no net export compensation
    """
    hour_of_day = hour_idx % 24
    period = classify_tou_basic(hour_of_day)
    rate = HOUR_TOU_RATES_BASIC[period]

    total_demand = house_kwh + ev_kwh

    if solar_kwh >= total_demand:
        leftover_solar = solar_kwh - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_kwh

    # Charge battery with leftover
    solar_unused = 0
    if leftover_solar > 0 and battery_state < battery_capacity:
        available_space = (battery_capacity - battery_state) / BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery = min(leftover_solar, available_space)
        battery_state += to_battery * BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar -= to_battery
        solar_unused = leftover_solar
    else:
        solar_unused = leftover_solar

    # Discharge battery
    if total_demand > 0 and battery_state > 0:
        discharge = min(total_demand, battery_state)
        total_demand -= discharge
        battery_state -= discharge

    # remainder from grid
    grid_kwh = total_demand
    cost = grid_kwh * rate

    return battery_state, grid_kwh, cost, solar_unused

def run_basic_hourly_sim(
    daily_house,
    daily_solar,
    daily_ev,
    battery_capacity=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
    """
    Basic hourly simulation for a year, naive battery usage, 
    no net export compensation, single import rates.
    """
    # 24-hour shapes for distributing daily usage
    house_shape = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02
    ])
    house_shape /= house_shape.sum()

    solar_shape = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10,
        0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()

    if ev_charging_pattern == "Night":
        ev_shape = np.array([
            0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    else:
        ev_shape = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15,
            0.15, 0.15, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    if ev_shape.sum() > 0:
        ev_shape /= ev_shape.sum()

    battery_state = 0.0
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
        "solar_unused": []
    }

    # Loop over 365 days
    for d in range(DAYS_PER_YEAR):
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
            h_h = h_24[hour]
            h_s = s_24[hour]
            h_e = e_24[hour]

            battery_state, grid_kwh, cost, sol_un = simulate_hour_basic(
                hour_idx, h_s, h_h, h_e, battery_state, battery_capacity
            )
            total_cost += cost
            total_grid += grid_kwh
            total_solar_unused += sol_un

            # log
            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(h_h)
            results["ev_kwh"].append(h_e)
            results["solar_kwh"].append(h_s)
            results["grid_kwh"].append(grid_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(sol_un)

    df = pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df

# --------------------------------------------------------------------------------
#             ADVANCED HOURLY "NEM 3.0-LIKE" MODEL FUNCTIONS
# --------------------------------------------------------------------------------

def classify_tou_adv(hour_of_day):
    if hour_of_day in ADV_TOU_SCHEDULE["on_peak_hours"]:
        return "on_peak"
    elif hour_of_day in ADV_TOU_SCHEDULE["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def any_future_on_peak(day, hour):
    # Check if any upcoming hour this day is on-peak
    for h in range(hour+1, 24):
        if h in ADV_TOU_SCHEDULE["on_peak_hours"]:
            return True
    return False

def advanced_battery_dispatch(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
    """
    1) use solar to meet load
    2) leftover -> battery
    3) if still leftover, export
    4) discharge battery if on-peak or if no future on-peak remains
    5) remainder from grid
    6) track export credit
    """
    day = hour_idx // 24
    hour_of_day = hour_idx % 24
    period = classify_tou_adv(hour_of_day)

    import_rate = IMPORT_RATES[period]
    export_rate = EXPORT_RATES[period]

    total_demand = house_kwh + ev_kwh

    # 1) solar offsets load
    if solar_kwh >= total_demand:
        leftover_solar = solar_kwh - total_demand
        total_demand = 0
    else:
        leftover_solar = 0
        total_demand -= solar_kwh

    # 2) charge battery
    export_kwh = 0
    if leftover_solar > 0:
        space_needed = (battery_capacity - battery_state) / ADV_BATTERY_EFFICIENCY
        if space_needed > 0:
            can_store = min(leftover_solar, space_needed)
            battery_state += can_store * ADV_BATTERY_EFFICIENCY
            leftover_solar -= can_store

        # 3) leftover solar is exported
        if leftover_solar > 0:
            export_kwh = leftover_solar
            leftover_solar = 0

    # 4) discharge battery
    on_peak = (hour_of_day in ADV_TOU_SCHEDULE["on_peak_hours"])
    if total_demand > 0 and battery_state > 0:
        if on_peak:
            discharge = min(total_demand, battery_state)
            total_demand -= discharge
            battery_state -= discharge
        else:
            if not any_future_on_peak(day, hour_of_day):
                discharge = min(total_demand, battery_state)
                total_demand -= discharge
                battery_state -= discharge
            # else do not discharge, save for upcoming on-peak

    # 5) remainder from grid
    grid_kwh = total_demand
    import_cost = grid_kwh * import_rate

    # 6) export credit
    export_credit = export_kwh * export_rate

    return battery_state, grid_kwh, import_cost, export_kwh, export_credit

def run_advanced_hourly_sim(
    daily_house,
    daily_solar,
    daily_ev,
    battery_capacity=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
    """
    365-day advanced hourly sim with separate import/export rates, 
    saving battery for on-peak.
    Returns (total_net_cost, df_hourly).
    """
    # 24-hour shapes
    house_shape = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.06,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07,
        0.08, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02
    ])
    house_shape /= house_shape.sum()

    solar_shape = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10,
        0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    if solar_shape.sum() > 0:
        solar_shape /= solar_shape.sum()

    if ev_charging_pattern == "Night":
        ev_shape = np.array([
            0.3, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    else:
        ev_shape = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15,
            0.15, 0.15, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    if ev_shape.sum() > 0:
        ev_shape /= ev_shape.sum()

    battery_state = 0.0
    total_import_cost = 0.0
    total_export_credit = 0.0

    results = {
        "day": [],
        "hour": [],
        "house_kwh": [],
        "ev_kwh": [],
        "solar_kwh": [],
        "grid_import_kwh": [],
        "import_cost": [],
        "export_kwh": [],
        "export_credit": [],
        "battery_state": []
    }

    for d in range(DAYS_PER_YEAR):
        if reset_battery_daily:
            battery_state = 0.0

        dh = daily_house[d]
        ds = daily_solar[d]
        de = daily_ev[d]

        hh_24 = dh * house_shape
        ss_24 = ds * solar_shape
        ee_24 = de * ev_shape

        for hour in range(24):
            hour_idx = d*24 + hour
            h_h = hh_24[hour]
            h_s = ss_24[hour]
            h_e = ee_24[hour]

            battery_state, g_kwh, imp_cost, exp_kwh, exp_credit = advanced_battery_dispatch(
                hour_idx, h_s, h_h, h_e, battery_state, battery_capacity
            )

            total_import_cost += imp_cost
            total_export_credit += exp_credit

            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(h_h)
            results["ev_kwh"].append(h_e)
            results["solar_kwh"].append(h_s)
            results["grid_import_kwh"].append(g_kwh)
            results["import_cost"].append(imp_cost)
            results["export_kwh"].append(exp_kwh)
            results["export_credit"].append(exp_credit)
            results["battery_state"].append(battery_state)

    df = pd.DataFrame(results)
    net_cost = total_import_cost - total_export_credit
    return net_cost, df

# --------------------------------------------------------------------------------
#                             STREAMLIT APP
# --------------------------------------------------------------------------------

def main():
    st.title("All-in-One: Monthly vs Basic Hourly vs Advanced Hourly (NEM 3.0) + Seasonality")

    st.write("""
    This single app demonstrates three distinct models, plus **optional seasonal variation** 
    for the **hourly** approaches.  
    - **Monthly** approach: Simple net-metering logic (NEM 2 vs NEM 3 naive).
    - **Basic Hourly**: Single import TOU rates, naive battery usage, no export credit.
    - **Advanced Hourly (NEM 3.0-like)**: Separate import/export rates, battery saved for on-peak.
    - **Seasonality** for hourly: Each month can scale daily solar/house usage with monthly factors.
    """)

    # ~~~~~ SIDEBAR ~~~~~
    st.sidebar.header("Common Inputs")

    # EV
    commute_miles = st.sidebar.slider("Daily Commute (miles)", 10, 100, DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days_option = st.sidebar.radio("EV Charging Frequency", ["Daily", "Weekdays Only"])
    days_per_week = 5 if charging_days_option == "Weekdays Only" else 7

    # For monthly model's battery logic
    monthly_charging_time = st.sidebar.radio("EV Charging Time (Monthly)", 
                                             ["Night (Super Off-Peak)", "Daytime (Peak)"])

    # Household
    household_consumption = st.sidebar.slider("Avg Household (kWh/day)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluctuation = st.sidebar.slider("Consumption Fluctuation (%)", 0, 50, 
                                    int(DEFAULT_CONSUMPTION_FLUCTUATION*100)) / 100

    # Solar & Battery
    solar_size = st.sidebar.slider("Solar Size (kW)", 3, 15, int(DEFAULT_SOLAR_SIZE))
    battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 0, 20, int(DEFAULT_BATTERY_CAPACITY))

    # Hourly specifics
    ev_charging_hourly = st.sidebar.selectbox("EV Charging Pattern (Hourly)", ["Night", "Daytime"])
    reset_battery_daily = st.sidebar.checkbox("Reset Battery Daily? (Hourly)", False)

    # ~~~~ Seasonal Variation Toggle for Hourly ~~~~
    use_seasonal_variation = st.sidebar.checkbox("Use Seasonal Variation (Hourly)?", False)
    st.sidebar.markdown("""
    **If enabled**, the hourly models will scale daily solar & household usage by month-specific factors.
    """)

    # Example monthly scaling factors for house load & solar
    # (Here we provide some arbitrary sample values. You can customize or add user sliders.)
    default_house_factors = [1.0, 0.95, 0.9, 0.9, 0.95, 1.0, 1.05, 1.05, 1.0, 1.0, 1.0, 1.0]
    default_solar_factors = [0.7, 0.75, 0.9, 1.0, 1.2, 1.3, 1.4, 1.35, 1.1, 0.9, 0.75, 0.65]

    # ---------- TABS ----------
    tab1, tab2, tab3 = st.tabs(["Monthly Approach", "Basic Hourly", "Advanced Hourly (NEM 3.0)"])

    # ================
    # TAB 1: Monthly
    # ================
    with tab1:
        st.header("Monthly Net Calculation")

        # 1) EV Demand
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)

        # 2) Household
        daily_house = household_consumption * (1 + fluctuation)
        house_monthly = calculate_monthly_values(daily_house)

        # 3) Solar
        _, solar_monthly = calculate_solar_production(solar_size)

        # 4) Run monthly cost calc
        (
            ev_cost_ns, ev_cost_n2, ev_cost_n3,
            total_ns, total_n2, total_n3
        ) = calculate_monthly_costs(
            ev_monthly, solar_monthly, house_monthly,
            battery_capacity,
            monthly_charging_time
        )

        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df_monthly = pd.DataFrame({
            "Month": months,
            "EV (kWh)": ev_monthly,
            "House (kWh)": house_monthly,
            "Solar (kWh)": solar_monthly,
            "EV Cost (No Solar)": ev_cost_ns,
            "EV Cost (NEM 2)": ev_cost_n2,
            "EV Cost (NEM 3+Batt)": ev_cost_n3,
            "Total (No Solar)": total_ns,
            "Total (NEM 2)": total_n2,
            "Total (NEM 3+Batt)": total_n3
        })

        st.subheader("Monthly Results Table")
        st.dataframe(df_monthly.style.format(precision=2))

        st.write("### Annual Summaries")
        st.write(f"**Annual EV Consumption**: {sum(ev_monthly):.1f} kWh")
        st.write(f"**Annual Household Consumption**: {sum(house_monthly):.1f} kWh")

        st.write(f"**Annual Solar Production**: {sum(solar_monthly):.1f} kWh")
        st.write(f"**Total Cost (No Solar)**: ${sum(total_ns):.2f}")
        st.write(f"**Total Cost (NEM 2)**: ${sum(total_n2):.2f}")
        st.write(f"**Total Cost (NEM 3+Batt)**: ${sum(total_n3):.2f}")

        # Plots
        st.write("### Monthly Consumption")
        fig1, ax1 = plt.subplots()
        ax1.bar(months, df_monthly["EV (kWh)"], label="EV")
        ax1.bar(months, df_monthly["House (kWh)"], bottom=df_monthly["EV (kWh)"], label="House")
        ax1.set_ylabel("kWh")
        ax1.legend()
        st.pyplot(fig1)

        st.write("### Solar Production")
        fig2, ax2 = plt.subplots()
        ax2.plot(months, df_monthly["Solar (kWh)"], marker="o", color="gold", label="Solar")
        ax2.set_ylabel("kWh")
        ax2.legend()
        st.pyplot(fig2)

        st.write("### EV Charging Cost Comparison")
        fig3, ax3 = plt.subplots()
        ax3.plot(months, df_monthly["EV Cost (No Solar)"], label="No Solar")
        ax3.plot(months, df_monthly["EV Cost (NEM 2)"], label="NEM 2")
        ax3.plot(months, df_monthly["EV Cost (NEM 3+Batt)"], label="NEM 3+Batt")
        ax3.set_ylabel("Cost ($)")
        ax3.legend()
        st.pyplot(fig3)

        st.write("### Total Monthly Costs")
        fig4, ax4 = plt.subplots()
        ax4.plot(months, df_monthly["Total (No Solar)"], label="No Solar")
        ax4.plot(months, df_monthly["Total (NEM 2)"], label="NEM 2")
        ax4.plot(months, df_monthly["Total (NEM 3+Batt)"], label="NEM 3+Batt")
        ax4.set_ylabel("Cost ($)")
        ax4.legend()
        st.pyplot(fig4)

        st.write("""
        **Notes (Monthly Approach)**  
        - Simplified net metering logic (credits at off-peak for NEM 2).  
        - Battery usage is aggregated monthly (a big approximation).  
        - Real NEM 3.0 typically involves **hourly** export rates, etc.
        """)

    # ================
    # TAB 2: Basic Hourly
    # ================
    with tab2:
        st.header("Basic Hourly Approach")

        if use_seasonal_variation:
            st.markdown("**Seasonal Variation Enabled**: We'll use monthly factors for house/solar in the daily arrays.")
            # Build daily arrays with seasonal scaling
            daily_house_kwh = build_daily_array_seasonal(
                household_consumption * (1+fluctuation), 
                default_house_factors
            )
            daily_solar_kwh = build_daily_array_seasonal(
                solar_size * 4,  # baseline 4 kWh/kW/day
                default_solar_factors
            )
        else:
            # No seasonality: same daily values all year
            daily_house_kwh = np.full(DAYS_PER_YEAR, household_consumption * (1+fluctuation))
            daily_solar_kwh = np.full(DAYS_PER_YEAR, solar_size * 4)

        # EV
        daily_ev_kwh = np.full(
            DAYS_PER_YEAR,
            (commute_miles / efficiency) * (days_per_week / 7.0)
        )

        cost_basic, grid_basic, solar_unused_basic, df_basic = run_basic_hourly_sim(
            daily_house_kwh,
            daily_solar_kwh,
            daily_ev_kwh,
            battery_capacity=battery_capacity,
            ev_charging_pattern=ev_charging_hourly,
            reset_battery_daily=reset_battery_daily
        )

        st.write("### Annual Results (Basic Hourly)")
        st.write(f"**Total Annual Cost:** ${cost_basic:,.2f}")
        st.write(f"**Total Grid Usage:** {grid_basic:,.0f} kWh")
        st.write(f"**Unused Solar:** {solar_unused_basic:,.0f} kWh (no export compensation)")

        st.write(f"**Avg Daily Cost:** ${cost_basic/DAYS_PER_YEAR:,.2f}")
        st.write(f"**Avg Daily Grid:** {grid_basic/DAYS_PER_YEAR:,.1f} kWh")
        st.write(f"**Avg Daily Unused Solar:** {solar_unused_basic/DAYS_PER_YEAR:,.1f} kWh")

        st.write("""
        **Notes (Basic Hourly)**  
        - Naive battery dispatch: use battery whenever there's load, charge with leftover solar.  
        - **No net export credit**: leftover solar is simply "lost."  
        - Single set of import rates (on/off/super-off-peak) for the day.
        """)

        # Let user pick a day to plot
        day_select = st.slider("Select a Day (0-364) for Plot", 0, DAYS_PER_YEAR-1, 0)
        df_day = df_basic[df_basic["day"] == day_select]

        st.write(f"### Hourly Profile - Day {day_select}")
        figA, axA = plt.subplots()
        axA.plot(df_day["hour"], df_day["house_kwh"], label="House Load")
        axA.plot(df_day["hour"], df_day["ev_kwh"], label="EV Load", linestyle="--")
        axA.plot(df_day["hour"], df_day["solar_kwh"], label="Solar", color="gold")
        axA.set_xlabel("Hour of Day")
        axA.set_ylabel("kWh")
        axA.legend()
        st.pyplot(figA)

        st.write("### Battery State and Grid Usage")
        figB, axB = plt.subplots()
        axB.plot(df_day["hour"], df_day["battery_state"], color="green", label="Battery (kWh)")
        axB.set_xlabel("Hour of Day")
        axB.set_ylabel("Battery (kWh)")
        axB.legend(loc="upper left")

        axC = axB.twinx()
        axC.plot(df_day["hour"], df_day["grid_kwh"], color="red", label="Grid Import (kWh)")
        axC.set_ylabel("Grid (kWh)")
        axC.legend(loc="upper right")

        st.pyplot(figB)

    # ================
    # TAB 3: Advanced Hourly (NEM 3.0)
    # ================
    with tab3:
        st.header("Advanced Hourly NEM 3.0-Like Approach")

        if use_seasonal_variation:
            st.markdown("**Seasonal Variation Enabled** for advanced hourly.")
            daily_house_kwh = build_daily_array_seasonal(
                household_consumption * (1+fluctuation),
                default_house_factors
            )
            daily_solar_kwh = build_daily_array_seasonal(
                solar_size * 4,
                default_solar_factors
            )
        else:
            daily_house_kwh = np.full(DAYS_PER_YEAR, household_consumption * (1+fluctuation))
            daily_solar_kwh = np.full(DAYS_PER_YEAR, solar_size * 4)

        daily_ev_kwh = np.full(
            DAYS_PER_YEAR,
            (commute_miles / efficiency) * (days_per_week / 7.0)
        )

        net_cost, df_adv = run_advanced_hourly_sim(
            daily_house_kwh,
            daily_solar_kwh,
            daily_ev_kwh,
            battery_capacity=battery_capacity,
            ev_charging_pattern=ev_charging_hourly,
            reset_battery_daily=reset_battery_daily
        )

        # Summaries
        st.write("### Annual Results (Advanced Hourly NEM 3.0-like)")
        st.write(f"**Total Annual Net Cost:** ${net_cost:,.2f}")

        total_import = df_adv["grid_import_kwh"].sum()
        total_export = df_adv["export_kwh"].sum()
        total_import_cost = df_adv["import_cost"].sum()
        total_export_credit = df_adv["export_credit"].sum()

        st.write(f"**Grid Imports:** {total_import:,.1f} kWh")
        st.write(f"**Solar Exports:** {total_export:,.1f} kWh")
        st.write(f"**Import Cost:** ${total_import_cost:,.2f}")
        st.write(f"**Export Credit:** -${total_export_credit:,.2f}")

        st.write(f"**Avg Daily Net Cost:** ${net_cost/DAYS_PER_YEAR:,.2f}")

        st.write("""
        **Notes (Advanced Hourly NEM 3.0)**  
        - Separate **import** vs. **export** rates, typically exporting for less credit than import cost.  
        - Battery is **saved** for on-peak if future on-peak hours remain that day.  
        - **Excess solar** is exported only if battery is full.  
        - Real NEM 3.0 can have monthly or hourly variations in export rates beyond this example.
        """)

        # Plot a selected day
        day2_select = st.slider("Select a Day (0-364) for Detailed Plot (Advanced)", 0, DAYS_PER_YEAR-1, 0)
        df_day2 = df_adv[df_adv["day"] == day2_select]

        st.write(f"### Hourly Profile - Day {day2_select}")
        figX, axX = plt.subplots()
        axX.plot(df_day2["hour"], df_day2["house_kwh"], label="House kWh")
        axX.plot(df_day2["hour"], df_day2["ev_kwh"], label="EV kWh", linestyle="--")
        axX.plot(df_day2["hour"], df_day2["solar_kwh"], label="Solar kWh", color="gold")
        axX.set_xlabel("Hour")
        axX.set_ylabel("kWh")
        axX.legend()
        st.pyplot(figX)

        st.write("### Battery, Import, and Export (Day View)")
        figY, axY = plt.subplots()
        axY.plot(df_day2["hour"], df_day2["battery_state"], label="Battery (kWh)", color="green")
        axY.set_xlabel("Hour")
        axY.set_ylabel("Battery (kWh)")
        axY.legend(loc="upper left")

        axZ = axY.twinx()
        axZ.plot(df_day2["hour"], df_day2["grid_import_kwh"], label="Grid Import", color="red")
        axZ.plot(df_day2["hour"], df_day2["export_kwh"], label="Solar Export", color="blue")
        axZ.set_ylabel("kWh")
        axZ.legend(loc="upper right")
        st.pyplot(figY)


if __name__ == "__main__":
    main()
