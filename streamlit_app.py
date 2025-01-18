import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For the advanced LP approach
try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
except ImportError:
    # If PuLP isn't installed, we can try to install it on the fly or display a warning.
    st.warning("PuLP not found. The 'Advanced Hourly LP' tab may not work unless PuLP is installed.")
    # Alternatively: !pip install pulp

# --------------------------------------------------------------------------------
#                                GLOBAL CONSTANTS
# --------------------------------------------------------------------------------

# ~~~~~ MONTHLY MODEL CONSTANTS ~~~~~
DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10  # kWh
DEFAULT_BATTERY_EFFICIENCY = 0.9  # 90%
DEFAULT_SOLAR_SIZE = 7.5  # kW
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8  # kWh/day
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2  # 20%
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SUMMER_MONTHS = [5, 6, 7, 8]  # June(5) - Sept(8)
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]  # Jan-May, Oct-Dec

# ~~~~~ Basic Hourly Model ~~~~~
HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16, 21)),       # 4 PM - 8 PM
    "off_peak_hours": list(range(7, 16)) + [21, 22],  # 7 AM - 3 PM, 9 PM - 10 PM
    "super_off_peak_hours": list(range(0, 7)) + [23], # 0 AM - 6 AM, 11 PM
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90

# ~~~~~ Advanced Hourly LP Approach (Seasonal, Weekend/Weekday) ~~~~~
DAYS_PER_YEAR = 365

# Example monthly scale factors for **local seasonality** (you can adjust in the sidebar)
DEFAULT_SOLAR_FACTORS = [0.6, 0.65, 0.75, 0.90, 1.0, 1.2, 1.3, 1.25, 1.0, 0.8, 0.65, 0.55]
DEFAULT_LOAD_FACTORS  = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.3, 1.1, 1.0, 1.0, 1.1]

# --------------------------------------------------------------------------------
#                      HELPER FUNCTIONS: MONTHLY MODEL
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    """Convert a constant daily_value into a monthly list by multiplying by days in each month."""
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
    """4 kWh/kW/day assumption."""
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
        # Simple "summer" vs. "winter" logic
        if month in SUMMER_MONTHS:
            on_peak = 0.45; off_peak = 0.25; super_off = 0.12
        else:
            on_peak = 0.35; off_peak = 0.20; super_off = 0.10

        # 1) No Solar
        cost_house_ns = household_monthly[month] * off_peak
        cost_ev_ns = ev_monthly[month] * super_off
        ev_cost_no_solar.append(cost_ev_ns)
        total_no_solar.append(cost_house_ns + cost_ev_ns)

        # 2) NEM 2.0
        excess_solar = solar_monthly[month] - household_monthly[month]
        credit_nem2 = max(0, excess_solar * off_peak)
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
            cost_ev_nem3 = ev_shortfall * super_off
        else:
            cost_ev_nem3 = ev_shortfall * on_peak

        ev_cost_nem_3.append(cost_ev_nem3)
        total_nem_3.append(cost_house_ns + cost_ev_nem3)

    return ev_cost_no_solar, ev_cost_nem_2, ev_cost_nem_3, total_no_solar, total_nem_2, total_nem_3

# --------------------------------------------------------------------------------
#               HELPER FUNCTIONS: BASIC HOURLY MODEL
# --------------------------------------------------------------------------------

def classify_tou_basic(hour_of_day):
    if hour_of_day in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour_of_day in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
    """Naive battery usage, no export credit, single set of import rates."""
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
    no net export compensation, single TOU rates.
    """
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
        # "Daytime" pattern example
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

            battery_state, g_kwh, cost, sol_un = simulate_hour_basic(
                hour_idx, h_s, h_h, h_e, battery_state, battery_capacity
            )
            total_cost += cost
            total_grid += g_kwh
            total_solar_unused += sol_un

            # log
            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(h_h)
            results["ev_kwh"].append(h_e)
            results["solar_kwh"].append(h_s)
            results["grid_kwh"].append(g_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(sol_un)

    df = pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df

# --------------------------------------------------------------------------------
#         ADVANCED HOURLY LP APPROACH: SEASONAL, WEEKDAY/WEEKEND, PARTIAL EV
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    """
    Synthetic schedule with columns [month, day_type, hour, import_rate, export_rate, demand_rate].
    Real data might come from a CSV or official tariff.
    """
    data = []
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                # E.g. on-peak from 4pm to 8pm
                if 16 <= h < 20:
                    import_r = 0.45
                    export_r = 0.12
                    demand_rate = 10.0 if d_type == "weekday" else 8.0
                elif 7 <= h < 16 or 20 <= h < 22:
                    import_r = 0.25
                    export_r = 0.08
                    demand_rate = 5.0
                else:
                    import_r = 0.12
                    export_r = 0.04
                    demand_rate = 2.0
                data.append({
                    "month": m,
                    "day_type": d_type,
                    "hour": h,
                    "import_rate": import_r,
                    "export_rate": export_r,
                    "demand_rate": demand_rate
                })
    df = pd.DataFrame(data)
    return df

def build_daily_arrays_with_factors(base_house_kwh, base_solar_kw, house_factors, solar_factors):
    """
    Create 365-day arrays for house load & solar production, scaled monthly by house_factors & solar_factors.
    Each list must be length 12. We multiply base_house_kwh by house_factors[m], etc.
    """
    daily_house = []
    daily_solar = []
    day_count = 0
    for m, ndays in enumerate(DAYS_IN_MONTH):
        for _ in range(ndays):
            # House scaled
            house_val = base_house_kwh * house_factors[m]
            # Solar scaled
            sol_val = (base_solar_kw * 4) * solar_factors[m]  # 4 kWh/kW/day baseline
            daily_house.append(house_val)
            daily_solar.append(sol_val)
            day_count += 1
            if day_count >= DAYS_PER_YEAR:
                break
        if day_count >= DAYS_PER_YEAR:
            break
    # Make sure we have exactly 365
    daily_house = daily_house[:DAYS_PER_YEAR]
    daily_solar = daily_solar[:DAYS_PER_YEAR]
    return np.array(daily_house), np.array(daily_solar)

def build_daily_ev_profile(daily_miles_mean=30, daily_miles_std=5, ev_eff=4.0,
                           ev_battery_cap=50.0):
    """
    Build a 365-array of EV kWh needed each day, e.g. random daily miles.
    We'll let the LP decide *which hours* to charge.
    """
    rng = np.random.default_rng(42)
    daily_ev = []
    for _ in range(DAYS_PER_YEAR):
        miles = rng.normal(daily_miles_mean, daily_miles_std)
        miles = max(0, miles)
        needed_kwh = miles / ev_eff
        needed_kwh = min(needed_kwh, ev_battery_cap)  # can't exceed battery capacity
        daily_ev.append(needed_kwh)
    return np.array(daily_ev)

# -------------- LP logic for each day --------------
def optimize_daily(
    day_idx,
    house_24,
    solar_24,
    ev_needed_kwh,
    ev_arrival,
    ev_depart,
    start_batt_soc,
    battery_cap_home,
    ev_battery_cap,  # not strictly used in constraints, but might be used if partial
    df_rates_day,
    demand_charge_enabled=False
):
    """
    Solve a daily LP for day_idx:
    - house_24: array(24) of house load
    - solar_24: array(24) of solar production
    - ev_needed_kwh: total kWh the EV must get this day
    - ev_arrival, ev_depart: hours (0-24)
    - start_batt_soc: home battery state at start
    - battery_cap_home: home battery capacity
    - df_rates_day: has columns [hour, import_rate, export_rate, demand_rate]
    - demand_charge_enabled: if True, we add demand charge logic

    Return (day_cost, end_batt_soc, df_day_solution)
    """
    prob = LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    # Decision variables
    home_batt_in  = LpVariable.dicts("home_batt_in", range(24), lowBound=0)
    home_batt_out = LpVariable.dicts("home_batt_out", range(24), lowBound=0)
    ev_charge     = LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc = [LpVariable(f"soc_{h}", lowBound=0, upBound=battery_cap_home) for h in range(25)]

    # Demand charge variable
    peak_demand = LpVariable("peak_demand", lowBound=0)

    cost_import = []
    credit_export = []

    for h in range(24):
        import_r = df_rates_day.loc[h, "import_rate"]
        export_r = df_rates_day.loc[h, "export_rate"]
        # demand_r = df_rates_day.loc[h, "demand_rate"] # used to define cost if needed

        # If outside EV window, no EV charge
        if not(ev_arrival <= h < ev_depart):
            prob += ev_charge[h] == 0, f"EV_cant_charge_{h}"

        # 1) Balance
        # solar_24[h] + home_batt_out[h] + grid_import[h] = 
        #     house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
        prob += (
            solar_24[h] + home_batt_out[h] + grid_import[h]
            == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
        ), f"Balance_{h}"

        # 2) Battery SOC recursion: soc[h+1] = soc[h] + batt_in - batt_out
        prob += (
            soc[h+1] == soc[h] + home_batt_in[h] - home_batt_out[h]
        ), f"SOC_{h}"

        # 3) Demand charge: peak_demand >= grid_import[h]
        if demand_charge_enabled:
            prob += peak_demand >= grid_import[h], f"Peak_dem_{h}"

        # cost terms
        cost_import.append(grid_import[h]*import_r)
        credit_export.append(grid_export[h]*export_r)

    # EV must get total needed for the day
    prob += sum(ev_charge[h] for h in range(24)) == ev_needed_kwh, "EV_requirement"

    # starting battery soc
    prob += soc[0] == start_batt_soc, "Start_batt"

    # objective
    total_import_cost = sum(cost_import)
    total_export_credit = sum(credit_export)
    if demand_charge_enabled:
        # We'll just multiply peak_demand by the max demand_rate for the day as a simple approach
        max_dem_rate = df_rates_day["demand_rate"].max()
        demand_cost = peak_demand * max_dem_rate
    else:
        demand_cost = 0
    prob.setObjective(total_import_cost - total_export_credit + demand_cost)

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    day_cost = value(prob.objective)
    end_batt_soc = value(soc[24])

    # Build day solution DataFrame
    results_hourly = []
    for h in range(24):
        row = {
            "hour": h,
            "grid_import": value(grid_import[h]),
            "grid_export": value(grid_export[h]),
            "batt_in": value(home_batt_in[h]),
            "batt_out": value(home_batt_out[h]),
            "ev_charge": value(ev_charge[h]),
        }
        results_hourly.append(row)
    df_day = pd.DataFrame(results_hourly)

    return day_cost, end_batt_soc, df_day


def run_advanced_lp_simulation(
    daily_house,
    daily_solar,
    daily_ev,
    ev_arrival_hr=18,
    ev_departure_hr=7,
    home_batt_capacity=10,
    ev_battery_cap=50,
    demand_charge_enabled=False
):
    """
    Multi-day approach: For each day, do a daily LP with day_type(weekday/weekend) & month from day index.
    We sum up costs, carry over end-of-day battery SOC as the next day's start.

    Returns total_cost, df_solution
    """
    df_rates = generate_utility_rate_schedule()

    # day_of_year -> (month, day_type)
    cum_days = np.cumsum([0]+DAYS_IN_MONTH)
    day_solutions = []
    total_cost = 0.0
    battery_soc = 0.0

    for day_idx in range(DAYS_PER_YEAR):
        # figure out month
        m = 0
        while m < 12 and day_idx >= cum_days[m+1]:
            m += 1

        # day_type
        dow = day_idx % 7
        if dow in [5,6]:
            d_type = "weekend"
        else:
            d_type = "weekday"

        # slice rates for this (month, day_type, hour)
        # then ensure it's sorted by hour 0-23
        df_rates_day = df_rates[(df_rates["month"]==m) & (df_rates["day_type"]==d_type)].copy()
        df_rates_day.sort_values("hour", inplace=True)
        df_rates_day.set_index("hour", inplace=True)

        # daily kWh
        day_house = daily_house[day_idx]
        day_solar = daily_solar[day_idx]
        day_ev = daily_ev[day_idx]

        # For simplicity, flatten house & solar into 24 lumps
        house_24 = np.full(24, day_house/24.0)
        solar_24 = np.full(24, day_solar/24.0)

        cost_day, end_soc, df_day = optimize_daily(
            day_idx,
            house_24,
            solar_24,
            day_ev,
            ev_arrival_hr,
            ev_departure_hr,
            battery_soc,
            home_batt_capacity,
            ev_battery_cap,
            df_rates_day,
            demand_charge_enabled=demand_charge_enabled
        )
        total_cost += cost_day
        battery_soc = end_soc

        df_day["day_idx"] = day_idx
        df_day["cost_day"] = cost_day
        df_day["month"] = m
        df_day["day_type"] = d_type
        day_solutions.append(df_day)

    df_solution = pd.concat(day_solutions, ignore_index=True)
    return total_cost, df_solution


# --------------------------------------------------------------------------------
#                           STREAMLIT MAIN APP
# --------------------------------------------------------------------------------
def main():
    st.title("All-In-One App: Monthly vs. Basic Hourly vs. Advanced LP with Seasonality & EV")

    st.write("""
    This single Streamlit app provides:
    1. **Monthly Net Approach** (simple NEM 2 vs. NEM 3 battery logic).
    2. **Basic Hourly** approach with naive battery usage, no export credit.
    3. **Advanced Hourly 'LP Optimization'** approach with:
       - Seasonal monthly factors for solar/house loads
       - Weekend vs. weekday TOU schedules
       - Partial EV charging (arrival/departure, random daily miles)
       - Demand charges
       - Hourly import/export rates
    """)

    st.sidebar.header("Common Inputs")

    # EV
    commute_miles = st.sidebar.slider("Daily Commute (miles)", 10, 100, DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_days_option = st.sidebar.radio("EV Charging Frequency (Monthly & Basic Hourly)", ["Daily", "Weekdays Only"])
    days_per_week = 5 if charging_days_option == "Weekdays Only" else 7

    monthly_time_of_charging = st.sidebar.radio("Monthly Model: EV Charging Time", 
                                               ["Night (Super Off-Peak)", "Daytime (Peak)"])

    # Household
    household_consumption = st.sidebar.slider("Base Daily Household (kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluctuation = st.sidebar.slider("Household Fluctuation (%)", 0, 50, int(DEFAULT_CONSUMPTION_FLUCTUATION*100)) / 100

    # Solar & Battery
    solar_size = st.sidebar.slider("Solar Size (kW)", 0, 15, int(DEFAULT_SOLAR_SIZE))
    battery_capacity = st.sidebar.slider("Battery Capacity (kWh) (Monthly & Basic Hourly)", 0, 20, int(DEFAULT_BATTERY_CAPACITY))

    # ~~~~~~~~~ Tabs ~~~~~~~~~
    tab1, tab2, tab3 = st.tabs(["Monthly Approach", "Basic Hourly", "Advanced Hourly LP"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     TAB 1: MONTHLY APPROACH
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab1:
        st.header("Monthly Net Calculation")

        # EV monthly
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)

        # House monthly
        daily_house_val = household_consumption * (1 + fluctuation)
        house_monthly = calculate_monthly_values(daily_house_val)

        # Solar monthly
        _, solar_monthly = calculate_solar_production(solar_size)

        # Run monthly costs
        (ev_ns, ev_n2, ev_n3,
         tot_ns, tot_n2, tot_n3) = calculate_monthly_costs(
            ev_monthly,
            solar_monthly,
            house_monthly,
            battery_capacity,
            monthly_time_of_charging
        )

        months = MONTH_NAMES
        df_m = pd.DataFrame({
            "Month": months,
            "EV (kWh)": ev_monthly,
            "House (kWh)": house_monthly,
            "Solar (kWh)": solar_monthly,
            "EV Cost (No Solar)": ev_ns,
            "EV Cost (NEM 2)": ev_n2,
            "EV Cost (NEM 3+Batt)": ev_n3,
            "Total (No Solar)": tot_ns,
            "Total (NEM 2)": tot_n2,
            "Total (NEM 3+Batt)": tot_n3
        })

        st.write("### Monthly Table")
        st.dataframe(df_m.style.format(precision=2))

        st.write("### Annual Summaries")
        st.write(f"**Annual EV kWh**: {sum(ev_monthly):.1f}")
        st.write(f"**Annual House kWh**: {sum(house_monthly):.1f}")
        st.write(f"**Annual Solar**: {sum(solar_monthly):.1f} kWh")

        st.write(f"**Total Cost (No Solar)**: ${sum(tot_ns):.2f}")
        st.write(f"**Total Cost (NEM 2)**: ${sum(tot_n2):.2f}")
        st.write(f"**Total Cost (NEM 3 + Batt)**: ${sum(tot_n3):.2f}")

        st.write("#### Visualization")
        fig1, ax1 = plt.subplots()
        ax1.bar(months, df_m["EV (kWh)"], label="EV")
        ax1.bar(months, df_m["House (kWh)"], bottom=df_m["EV (kWh)"], label="House")
        ax1.set_ylabel("kWh")
        ax1.set_title("Monthly Consumption")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(months, df_m["Solar (kWh)"], marker="o", color="gold", label="Solar")
        ax2.set_ylabel("kWh")
        ax2.set_title("Monthly Solar Production")
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(months, df_m["Total (No Solar)"], label="No Solar")
        ax3.plot(months, df_m["Total (NEM 2)"], label="NEM 2")
        ax3.plot(months, df_m["Total (NEM 3+Batt)"], label="NEM 3 + Batt")
        ax3.set_ylabel("Cost ($)")
        ax3.set_title("Monthly Total Costs")
        ax3.legend()
        st.pyplot(fig3)

        st.write("""
        **Notes**: This is a **simplified monthly** net metering approach. Real NEM 3.0 
        typically uses hourly export rates and more complex calculations.
        """)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                     TAB 2: BASIC HOURLY
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab2:
        st.header("Basic Hourly Approach (Naive Battery, No Export Credit)")

        # Build 365-day arrays (no seasonality toggle here, you can add if desired)
        daily_house_val = household_consumption * (1 + fluctuation)
        daily_house_arr = np.full(DAYS_PER_YEAR, daily_house_val)
        daily_solar_arr = np.full(DAYS_PER_YEAR, solar_size * 4)  # 4 kWh/kW/day
        daily_ev_arr = np.full(DAYS_PER_YEAR, (commute_miles / efficiency)*(days_per_week/7.0))

        reset_daily_batt = st.checkbox("Reset Battery Each Day? (Basic Hourly)", False)
        ev_pattern_basic = st.selectbox("EV Charging Pattern (Night/Daytime - Basic Hourly)",
                                        ["Night","Daytime"])

        cost_basic, grid_basic, solar_unused, df_basic = run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            battery_capacity=battery_capacity,
            ev_charging_pattern=ev_pattern_basic,
            reset_battery_daily=reset_daily_batt
        )

        st.write("### Annual Results")
        st.write(f"**Total Annual Cost**: ${cost_basic:,.2f}")
        st.write(f"**Grid Usage**: {grid_basic:,.0f} kWh")
        st.write(f"**Unused Solar**: {solar_unused:,.0f} kWh")

        # Show day-level or hour-level details
        day_sel = st.slider("Pick a Day (0-364) to see hourly breakdown", 0, DAYS_PER_YEAR-1, 0)
        df_day_sel = df_basic[df_basic["day"]==day_sel]

        st.write(f"### Hourly Data - Day {day_sel}")
        figA, axA = plt.subplots()
        axA.plot(df_day_sel["hour"], df_day_sel["house_kwh"], label="House")
        axA.plot(df_day_sel["hour"], df_day_sel["ev_kwh"], label="EV", linestyle="--")
        axA.plot(df_day_sel["hour"], df_day_sel["solar_kwh"], label="Solar", color="gold")
        axA.set_xlabel("Hour")
        axA.set_ylabel("kWh")
        axA.legend()
        st.pyplot(figA)

        st.write("Battery State & Grid Usage")
        figB, axB = plt.subplots()
        axB.plot(df_day_sel["hour"], df_day_sel["battery_state"], color="green", label="Battery (kWh)")
        axB.set_xlabel("Hour")
        axB.set_ylabel("Battery (kWh)")
        axB.legend(loc="upper left")

        axC = axB.twinx()
        axC.plot(df_day_sel["hour"], df_day_sel["grid_kwh"], color="red", label="Grid Import")
        axC.set_ylabel("Grid (kWh)")
        axC.legend(loc="upper right")
        st.pyplot(figB)

        st.write("""
        **Notes**: 
        - This approach uses a naive battery dispatch. 
        - **No** export credit: any excess solar is lost.
        - Simplified TOU: single on/off/super-off-peak set, 
          ignoring weekend vs. weekday or season.
        """)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                   TAB 3: ADVANCED HOURLY LP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab3:
        st.header("Advanced Hourly LP Approach")

        st.write("""
        **Features**:
        - Monthly scaling factors for house & solar (seasonality).
        - Weekend vs. weekday rate differences.
        - Partial EV charging (daily random miles, arrival/departure hours).
        - Demand charges (optional).
        - Real (synthetic) import/export rates each hour, 
          daily optimization for each day in the year.
        """)

        # Let user set up advanced parameters
        st.subheader("Seasonal Factors & Advanced EV / Demand Charge")

        # House & solar factors
        adv_house_factors = []
        adv_solar_factors = []
        with st.expander("Customize Monthly Scale Factors"):
            for i,mn in enumerate(MONTH_NAMES):
                hf = st.slider(f"{mn} House Factor", 0.5, 1.5, DEFAULT_LOAD_FACTORS[i], 0.05, key=f"housef_{i}")
                sf = st.slider(f"{mn} Solar Factor", 0.5, 1.5, DEFAULT_SOLAR_FACTORS[i], 0.05, key=f"solarf_{i}")
                adv_house_factors.append(hf)
                adv_solar_factors.append(sf)

        # EV daily random usage
        adv_ev_miles_mean = st.slider("Mean Daily Miles (Advanced EV)", 0, 100, 30)
        adv_ev_miles_std = st.slider("StdDev Daily Miles", 0, 30, 5)
        adv_ev_eff = st.slider("EV Efficiency (miles/kWh) - Advanced LP", 3.0, 5.0, 4.0)
        adv_ev_batt_cap = st.slider("EV Battery Capacity (kWh) for partial charges", 20, 100, 50)
        adv_ev_arrival = st.slider("EV Arrival Hour", 0, 23, 18)
        adv_ev_depart = st.slider("EV Departure Hour", 0, 23, 7)

        adv_batt_capacity_home = st.slider("Home Battery Capacity (kWh) - Advanced LP", 0, 40, 10)
        adv_demand_charges = st.checkbox("Enable Demand Charges in the LP?", False)

        st.write("**Running Advanced Simulation** (This might take a moment to solve 365 LPs)...")

        daily_house_advanced, daily_solar_advanced = build_daily_arrays_with_factors(
            household_consumption,  # base daily house
            solar_size,             # base solar kW
            adv_house_factors,
            adv_solar_factors
        )
        # Apply fluctuation
        daily_house_advanced *= (1 + fluctuation)

        daily_ev_advanced = build_daily_ev_profile(
            adv_ev_miles_mean, adv_ev_miles_std, adv_ev_eff, adv_ev_batt_cap
        )

        # Solve the multi-day LP
        if "pulp" in globals():  # confirm PuLP is installed
            adv_total_cost, df_adv_sol = run_advanced_lp_simulation(
                daily_house_advanced,
                daily_solar_advanced,
                daily_ev_advanced,
                ev_arrival_hr=adv_ev_arrival,
                ev_departure_hr=adv_ev_depart,
                home_batt_capacity=adv_batt_capacity_home,
                ev_battery_cap=adv_ev_batt_cap,
                demand_charge_enabled=adv_demand_charges
            )
            st.success(f"Advanced Simulation Complete. Total Annual Net Cost: ${adv_total_cost:,.2f}")

            # Summaries
            total_import_kwh = df_adv_sol["grid_import"].sum()
            total_export_kwh = df_adv_sol["grid_export"].sum()
            st.write(f"**Grid Imports**: {total_import_kwh:,.1f} kWh")
            st.write(f"**Solar Exports**: {total_export_kwh:,.1f} kWh")

            # Let user pick day
            day_pick = st.slider("Select a Day (0-364) to see LP results", 0, DAYS_PER_YEAR-1, 0)
            df_day_pick = df_adv_sol[df_adv_sol["day_idx"]==day_pick].copy()

            st.write(f"### Day {day_pick} Hourly Dispatch")
            figX, axX = plt.subplots()
            axX.plot(df_day_pick["hour"], df_day_pick["batt_in"], label="Battery In", color="orange")
            axX.plot(df_day_pick["hour"], df_day_pick["batt_out"], label="Battery Out", color="green")
            axX.plot(df_day_pick["hour"], df_day_pick["ev_charge"], label="EV Charge", color="red")
            axX.set_xlabel("Hour")
            axX.set_ylabel("kWh")
            axX.legend()
            st.pyplot(figX)

            st.write("""
            **Interpreting**:
            - The LP decides each hour how much to charge the home battery, 
              discharge it, or charge the EV, to minimize total cost 
              (import cost - export credit + demand charge).
            - If demand charges are enabled, it also tries to minimize peak demand.
            """)

        else:
            st.error("PuLP not available. Please install PuLP to run the Advanced Hourly LP model.")

if __name__ == "__main__":
    main()
