import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Attempt to import PuLP for the advanced LP approach
try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

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
SUMMER_MONTHS = [5, 6, 7, 8]  # June(5)-Sep(8)
WINTER_MONTHS = [0,1,2,3,4,9,10,11]

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

# ~~~~~ Advanced Hourly LP ~~~~~
DAYS_PER_YEAR = 365

# Example monthly scale factors for seasonality
DEFAULT_SOLAR_FACTORS = [0.6, 0.65, 0.75, 0.90, 1.0, 1.2, 1.3, 1.25, 1.0, 0.8, 0.65, 0.55]
DEFAULT_LOAD_FACTORS  = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.3, 1.1, 1.0, 1.0, 1.1]

# --------------------------------------------------------------------------------
#                           HELPER FUNCTIONS: MONTHLY
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    return [daily_value * d for d in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
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

def calculate_monthly_costs(ev_monthly, solar_monthly, house_monthly, battery_capacity, time_of_charging):
    ev_cost_ns, ev_cost_n2, ev_cost_n3 = [], [], []
    total_ns, total_n2, total_n3 = [], [], []

    battery_state = 0.0

    for m in range(12):
        # simple "summer" vs. "winter"
        if m in SUMMER_MONTHS:
            on_peak = 0.45; off_peak = 0.25; super_off = 0.12
        else:
            on_peak = 0.35; off_peak = 0.20; super_off = 0.10

        # 1) No solar
        cost_house_ns = house_monthly[m] * off_peak
        cost_ev_ns = ev_monthly[m] * super_off
        ev_cost_ns.append(cost_ev_ns)
        total_ns.append(cost_house_ns + cost_ev_ns)

        # 2) NEM 2.0
        excess_solar = solar_monthly[m] - house_monthly[m]
        credit_n2 = max(0, excess_solar * off_peak)
        cost_ev_n2 = max(0, ev_monthly[m] - credit_n2)
        ev_cost_n2.append(cost_ev_n2)
        total_n2.append(cost_house_ns - credit_n2 + cost_ev_n2)

        # 3) NEM 3 + battery (naive monthly)
        leftover_solar = max(0, solar_monthly[m] - house_monthly[m])
        ev_short = ev_monthly[m]

        if time_of_charging == "Daytime (Peak)" and leftover_solar>0:
            direct_solar = min(ev_short, leftover_solar)
            ev_short -= direct_solar
            leftover_solar -= direct_solar

        # charge battery
        if leftover_solar>0 and battery_state < battery_capacity:
            can_charge = min(leftover_solar, battery_capacity - battery_state)
            battery_state += can_charge * DEFAULT_BATTERY_EFFICIENCY
            leftover_solar -= can_charge

        # discharge battery
        if ev_short>0 and battery_state>0:
            discharge = min(ev_short, battery_state)
            ev_short -= discharge
            battery_state -= discharge

        # remainder from grid
        if time_of_charging == "Night (Super Off-Peak)":
            cost_ev_n3 = ev_short * super_off
        else:
            cost_ev_n3 = ev_short * on_peak

        ev_cost_n3.append(cost_ev_n3)
        total_n3.append(cost_house_ns + cost_ev_n3)

    return ev_cost_ns, ev_cost_n2, ev_cost_n3, total_ns, total_n2, total_n3


# --------------------------------------------------------------------------------
#              HELPER FUNCTIONS: BASIC HOURLY
# --------------------------------------------------------------------------------

def classify_tou_basic(hour):
    if hour in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
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

    solar_unused = 0
    if leftover_solar>0 and battery_state<battery_capacity:
        available_space = (battery_capacity - battery_state)/BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery = min(leftover_solar, available_space)
        battery_state += to_battery * BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar -= to_battery
        solar_unused = leftover_solar
    else:
        solar_unused = leftover_solar

    if total_demand>0 and battery_state>0:
        discharge = min(total_demand, battery_state)
        total_demand -= discharge
        battery_state -= discharge

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
    house_shape = np.array([
        0.02,0.02,0.02,0.02,0.02,0.03,0.04,0.06,
        0.06,0.06,0.06,0.05,0.05,0.05,0.06,0.07,
        0.08,0.06,0.05,0.05,0.04,0.03,0.02,0.02
    ])
    house_shape /= house_shape.sum()

    solar_shape = np.array([
        0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.10,
        0.15,0.20,0.20,0.15,0.10,0.05,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    ])
    if solar_shape.sum()>0:
        solar_shape /= solar_shape.sum()

    if ev_charging_pattern == "Night":
        ev_shape = np.array([
            0.3,0.3,0.3,0.1,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ])
    else:
        ev_shape = np.array([
            0.0,0.0,0.0,0.0,0.0,0.05,0.10,0.15,
            0.15,0.15,0.15,0.10,0.05,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ])
    if ev_shape.sum()>0:
        ev_shape /= ev_shape.sum()

    battery_state = 0.0
    total_cost = 0.0
    total_grid = 0.0
    total_solar_unused = 0.0

    results = {"day":[],"hour":[],"house_kwh":[],"ev_kwh":[],"solar_kwh":[],"grid_kwh":[],"cost":[],"battery_state":[],"solar_unused":[]}

    days = len(daily_house)
    for d in range(days):
        if reset_battery_daily:
            battery_state = 0.0

        dh = daily_house[d]
        ds = daily_solar[d]
        de = daily_ev[d]

        # Distribute daily data over 24 hours
        h_24 = dh * house_shape
        s_24 = ds * solar_shape
        e_24 = de * ev_shape

        for hour in range(24):
            hour_idx = d*24 + hour
            bh = h_24[hour]
            bs = s_24[hour]
            be = e_24[hour]

            battery_state, g_kwh, cost, sol_un = simulate_hour_basic(
                hour_idx, bs, bh, be, battery_state, battery_capacity
            )
            total_cost += cost
            total_grid += g_kwh
            total_solar_unused += sol_un

            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(bh)
            results["ev_kwh"].append(be)
            results["solar_kwh"].append(bs)
            results["grid_kwh"].append(g_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(sol_un)

    df = pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df

# --------------------------------------------------------------------------------
#          ADVANCED HOURLY LP WITH MULTIPLE MODES + TOGGLE FOR EXCESS SOLAR
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    """
    Synthetic schedule for month, day_type, hour => import_rate, export_rate, demand_rate
    """
    data = []
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    import_r = 0.45
                    export_r = 0.12
                    demand_r = 10.0 if d_type=="weekday" else 8.0
                elif 7<=h<16 or 20<=h<22:
                    import_r = 0.25
                    export_r = 0.08
                    demand_r = 5.0
                else:
                    import_r = 0.12
                    export_r = 0.04
                    demand_r = 2.0
                data.append({
                    "month": m,
                    "day_type": d_type,
                    "hour": h,
                    "import_rate": import_r,
                    "export_rate": export_r,
                    "demand_rate": demand_r
                })
    return pd.DataFrame(data)

def build_daily_arrays_with_factors(base_house_kwh, base_solar_kw, house_factors, solar_factors, fluct=0.0):
    daily_house = []
    daily_solar = []
    day_count = 0
    for m, ndays in enumerate(DAYS_IN_MONTH):
        for _ in range(ndays):
            hv = base_house_kwh * house_factors[m] * (1+fluct)
            sv = (base_solar_kw*4) * solar_factors[m]
            daily_house.append(hv)
            daily_solar.append(sv)
            day_count+=1
            if day_count>=DAYS_PER_YEAR:
                break
        if day_count>=DAYS_PER_YEAR:
            break
    daily_house = np.array(daily_house[:DAYS_PER_YEAR])
    daily_solar = np.array(daily_solar[:DAYS_PER_YEAR])
    return daily_house, daily_solar

def build_daily_ev_profile(daily_miles_mean=30, daily_miles_std=5, ev_eff=4.0, ev_batt_cap=50):
    rng = np.random.default_rng(42)
    arr = []
    for _ in range(DAYS_PER_YEAR):
        miles = rng.normal(daily_miles_mean, daily_miles_std)
        miles = max(0,miles)
        needed = miles/ev_eff
        needed = min(needed, ev_batt_cap)
        arr.append(needed)
    return np.array(arr)

def hour_to_ampm_label(h):
    """Convert 0-23 to a string like '12 AM','1 AM','...','12 PM','1 PM', etc."""
    if h==0: return "12 AM"
    elif h<12: return f"{h} AM"
    elif h==12: return "12 PM"
    else: return f"{h-12} PM"

def advanced_battery_constraints(prob, hour, hour_of_day,
                                 battery_mode, backup_reserve_frac,
                                 home_batt_out, grid_export, soc, batt_cap,
                                 self_consumption_excess="Curtail"):
    """
    Add constraints per hour for chosen battery_mode & leftover solar toggle:
      - "TOU Arbitrage": only discharge in on-peak (16–19).
      - "Self-Consumption": 
         * if self_consumption_excess=="Curtail" => no net export => grid_export[h] == 0
         * if self_consumption_excess=="Export"  => allow export
      - "Backup Priority": keep SOC >= fraction*batt_cap
      - "None": no special constraints
    """
    # 1) TOU Arbitrage
    if battery_mode=="TOU Arbitrage":
        if hour_of_day<16 or hour_of_day>=20:
            prob += home_batt_out[hour] == 0, f"TOUArb_offpeak_no_out_{hour}"
    # 2) Self-Consumption
    elif battery_mode=="Self-Consumption":
        if self_consumption_excess=="Curtail":
            # no net export
            prob += grid_export[hour] == 0, f"SelfCons_no_export_{hour}"
        else:
            # "Export leftover" => do not force grid_export=0
            pass
    # 3) Backup Priority
    elif battery_mode=="Backup Priority":
        prob += soc[hour] >= backup_reserve_frac*batt_cap, f"BackupSOC_{hour}"
    # 4) "None" => no additional constraints

def optimize_daily(
    day_idx, 
    house_24, 
    solar_24,
    ev_needed_kwh,
    ev_arrival,
    ev_depart,
    start_batt_soc,
    home_batt_capacity,
    ev_batt_capacity,
    df_rates_day,
    demand_charge_enabled=False,
    battery_mode="None",
    backup_reserve_frac=0.2,
    self_consumption_excess="Curtail"
):
    """
    Solve daily LP with multiple modes + leftover solar toggle for self-consumption.
    """
    if not PULP_AVAILABLE:
        return None, start_batt_soc, pd.DataFrame()

    prob = LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc = [pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=home_batt_capacity) for h in range(25)]

    peak_demand = pulp.LpVariable("peak_demand", lowBound=0)

    cost_import = []
    credit_export = []

    for h in range(24):
        import_r  = df_rates_day.loc[h,"import_rate"]
        export_r  = df_rates_day.loc[h,"export_rate"]

        # EV can only charge if h in [arrival, depart)
        if not (ev_arrival <= h < ev_depart):
            prob += ev_charge[h] == 0, f"EVNoCharge_{h}"

        # power balance
        prob += (
            solar_24[h] + home_batt_out[h] + grid_import[h]
            == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
        ), f"Balance_{h}"

        # battery SOC recursion
        prob += soc[h+1] == soc[h] + home_batt_in[h] - home_batt_out[h], f"SOC_{h}"

        # demand charge
        if demand_charge_enabled:
            prob += peak_demand >= grid_import[h], f"Peak_{h}"

        cost_import.append(grid_import[h]*import_r)
        credit_export.append(grid_export[h]*export_r)

        # Insert constraints for chosen mode
        advanced_battery_constraints(prob, h, h,
                                     battery_mode, backup_reserve_frac,
                                     home_batt_out, grid_export, soc, home_batt_capacity,
                                     self_consumption_excess)

    # EV daily total
    prob += sum(ev_charge[h] for h in range(24)) == ev_needed_kwh, "EVRequirement"

    # start battery
    prob += soc[0] == start_batt_soc, "StartSOC"

    # objective
    total_import_cost = sum(cost_import)
    total_export_credit= sum(credit_export)
    demand_cost = 0
    if demand_charge_enabled:
        max_dem_rate_day = df_rates_day["demand_rate"].max()
        demand_cost = peak_demand*max_dem_rate_day

    prob.setObjective(total_import_cost - total_export_credit + demand_cost)
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status]=="Optimal":
        day_cost = pulp.value(prob.objective)
        end_soc = pulp.value(soc[24])
    else:
        day_cost = None
        end_soc  = start_batt_soc

    # gather day solution
    hours_solution = []
    for h in range(24):
        row = {
            "hour": h,
            "grid_import": pulp.value(grid_import[h]),
            "grid_export": pulp.value(grid_export[h]),
            "batt_in": pulp.value(home_batt_in[h]),
            "batt_out": pulp.value(home_batt_out[h]),
            "ev_charge": pulp.value(ev_charge[h]),
        }
        hours_solution.append(row)
    df_day = pd.DataFrame(hours_solution)

    return day_cost, end_soc, df_day


def run_advanced_lp_sim(
    daily_house,
    daily_solar,
    daily_ev,
    ev_arrival_hr=18,
    ev_departure_hr=7,
    home_batt_capacity=10,
    ev_battery_cap=50,
    demand_charge_enabled=False,
    battery_mode="None",
    backup_reserve_frac=0.2,
    self_consumption_excess="Curtail"
):
    """
    Multi-day approach. For each day, solve daily LP. Sum cost, carry battery SOC.
    """
    if not PULP_AVAILABLE:
        return None, pd.DataFrame()

    df_rates = generate_utility_rate_schedule()
    cum_days = np.cumsum([0]+DAYS_IN_MONTH)
    day_solutions = []
    total_cost = 0.0
    battery_soc = 0.0

    for day_idx in range(DAYS_PER_YEAR):
        # figure out month
        m=0
        while m<12 and day_idx>=cum_days[m+1]:
            m+=1
        dow = day_idx%7
        d_type = "weekend" if dow in [5,6] else "weekday"

        df_day_rates = df_rates[(df_rates["month"]==m) & (df_rates["day_type"]==d_type)].copy()
        df_day_rates.sort_values("hour", inplace=True)
        df_day_rates.set_index("hour", inplace=True)

        dh = daily_house[day_idx]
        ds = daily_solar[day_idx]
        de = daily_ev[day_idx]

        house_24 = np.full(24, dh/24.0)
        solar_24 = np.full(24, ds/24.0)

        day_cost, end_soc, df_day_sol = optimize_daily(
            day_idx, house_24, solar_24, de,
            ev_arrival_hr, ev_departure_hr,
            battery_soc, home_batt_capacity, ev_battery_cap,
            df_day_rates, demand_charge_enabled,
            battery_mode, backup_reserve_frac,
            self_consumption_excess
        )
        if day_cost is not None:
            total_cost += day_cost
        battery_soc = end_soc

        df_day_sol["day_idx"] = day_idx
        df_day_sol["month"]   = m
        df_day_sol["day_type"]= d_type
        df_day_sol["cost_day"]= day_cost if day_cost else 0
        day_solutions.append(df_day_sol)

    df_sol = pd.concat(day_solutions, ignore_index=True)
    return total_cost, df_sol

# --------------------------------------------------------------------------------
#                          STREAMLIT MAIN APP
# --------------------------------------------------------------------------------

def main():
    st.title("App with Self-Consumption Toggle for Excess Solar + AM/PM & Multiple Battery Modes")

    st.write("""
    - **Self-Consumption** mode now has a **toggle** for **Excess Solar**:
      *Curtail leftover solar* vs. *Export leftover solar*.
    - All other modes remain the same.
    """)

    st.sidebar.header("Common Inputs")

    # EV basics
    commute_miles = st.sidebar.slider("Daily Commute (miles)",10,100, DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_freq = st.sidebar.radio("EV Charging Frequency (Monthly & Basic Hourly)", ["Daily","Weekdays Only"])
    days_per_week = 5 if charging_freq=="Weekdays Only" else 7

    monthly_charge_time = st.sidebar.radio("Monthly Model: EV Charging Time", ["Night (Super Off-Peak)","Daytime (Peak)"])

    # House
    house_kwh_base = st.sidebar.slider("Daily House (kWh)",10,50,int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluct = st.sidebar.slider("House Fluctuation (%)",0,50,int(DEFAULT_CONSUMPTION_FLUCTUATION*100))/100

    # Solar & Battery (for monthly/basic)
    solar_size = st.sidebar.slider("Solar Size (kW)",0,15,int(DEFAULT_SOLAR_SIZE))
    monthly_batt_capacity = st.sidebar.slider("Battery (kWh) for Monthly/Basic Hourly",0,20,int(DEFAULT_BATTERY_CAPACITY))

    # TABS
    tab1, tab2, tab3 = st.tabs(["Monthly Approach","Basic Hourly","Advanced Hourly LP (Modes)"])

    # ~~~~~~~~~ TAB 1: Monthly ~~~~~~~~~
    with tab1:
        st.header("Monthly Net Approach")
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)
        daily_house_val = house_kwh_base*(1+fluct)
        house_monthly = calculate_monthly_values(daily_house_val)
        _, solar_monthly= calculate_solar_production(solar_size)

        (ev_ns, ev_n2, ev_n3, tot_ns, tot_n2, tot_n3) = calculate_monthly_costs(
            ev_monthly, solar_monthly, house_monthly, monthly_batt_capacity, monthly_charge_time
        )

        df_m = pd.DataFrame({
            "Month": MONTH_NAMES,
            "EV (kWh)": ev_monthly,
            "House (kWh)": house_monthly,
            "Solar (kWh)": solar_monthly,
            "EV Cost (NoSolar)": ev_ns,
            "EV Cost (NEM2)": ev_n2,
            "EV Cost (NEM3+Batt)": ev_n3,
            "Total (NoSolar)": tot_ns,
            "Total (NEM2)": tot_n2,
            "Total (NEM3+Batt)": tot_n3
        })
        st.dataframe(df_m.style.format(precision=2))

        st.write("### Annual Summaries")
        st.write(f"Annual EV kWh: {sum(ev_monthly):.1f}")
        st.write(f"Annual House kWh: {sum(house_monthly):.1f}")
        st.write(f"Annual Solar kWh: {sum(solar_monthly):.1f}")
        st.write(f"Total (NoSolar): ${sum(tot_ns):.2f}")
        st.write(f"Total (NEM2):    ${sum(tot_n2):.2f}")
        st.write(f"Total (NEM3+Batt): ${sum(tot_n3):.2f}")

    # ~~~~~~~~~ TAB 2: Basic Hourly ~~~~~~~~~
    with tab2:
        st.header("Basic Hourly Approach")
        days = DAYS_PER_YEAR
        daily_house_arr = np.full(days, house_kwh_base*(1+fluct))
        daily_solar_arr = np.full(days, solar_size*4)
        daily_ev_arr    = np.full(days, (commute_miles/efficiency)*(days_per_week/7.0))

        reset_daily_batt = st.checkbox("Reset Battery Daily? (Basic Hourly)",False)
        ev_basic_pattern = st.selectbox("EV Charging Pattern (Night/Daytime)", ["Night","Daytime"])

        cost_b, grid_b, solar_un_b, df_b = run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            monthly_batt_capacity,
            ev_charging_pattern=ev_basic_pattern,
            reset_battery_daily=reset_daily_batt
        )

        st.write(f"**Total Annual Cost**: ${cost_b:,.2f}")
        st.write(f"**Grid Usage**: {grid_b:,.0f} kWh")
        st.write(f"**Unused Solar**: {solar_un_b:,.0f} kWh")

        day_select_b = st.slider("Pick a day (0-364)",0,364,0)
        df_day_b = df_b[df_b["day"]==day_select_b].copy()
        df_day_b["hour_label"] = df_day_b["hour"].apply(hour_to_ampm_label)

        st.write(f"### Hourly Data - Day {day_select_b}")
        figB, axB = plt.subplots()
        axB.plot(df_day_b["hour_label"], df_day_b["house_kwh"], label="House")
        axB.plot(df_day_b["hour_label"], df_day_b["ev_kwh"], label="EV", linestyle="--")
        axB.plot(df_day_b["hour_label"], df_day_b["solar_kwh"], label="Solar", color="gold")
        axB.set_ylabel("kWh")
        axB.set_xlabel("Hour")
        plt.xticks(rotation=45)
        axB.legend()
        st.pyplot(figB)

        st.write("Battery State & Grid Usage")
        figB2, axB2 = plt.subplots()
        axB2.plot(df_day_b["hour_label"], df_day_b["battery_state"], color="green", label="Battery (kWh)")
        axB2.set_xlabel("Hour")
        axB2.set_ylabel("Battery (kWh)")
        plt.xticks(rotation=45)
        axB2.legend(loc="upper left")

        axB3 = axB2.twinx()
        axB3.plot(df_day_b["hour_label"], df_day_b["grid_kwh"], color="red", label="Grid Import")
        axB3.set_ylabel("Grid (kWh)")
        axB3.legend(loc="upper right")
        st.pyplot(figB2)

    # ~~~~~~~~~ TAB 3: Advanced Hourly LP (Modes)
    with tab3:
        st.header("Advanced Hourly LP with Self-Consumption Toggle for Excess Solar")

        if not PULP_AVAILABLE:
            st.error("PuLP not available. Install `pip install pulp` for this tab.")
        else:
            st.write("""
            **Battery Modes**:
            - **None**: unconstrained cost optimization
            - **TOU Arbitrage**: only discharge on-peak (4–8 PM)
            - **Self-Consumption**: 
               - Toggle how leftover solar is handled: 'Curtail' (no export) or 'Export leftover'
            - **Backup Priority**: keep a fraction of battery in reserve
            """)

            adv_batt_mode = st.selectbox("Battery Mode", ["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_reserve_slider = 0.0
            self_consumption_excess_handling = "Curtail"
            if adv_batt_mode == "Backup Priority":
                backup_reserve_slider = st.slider("Backup Reserve (%)",0,50,20)/100
            elif adv_batt_mode == "Self-Consumption":
                # Provide a toggle for leftover solar
                self_consumption_excess_handling = st.radio("Excess Solar in Self-Consumption?", ["Curtail","Export"])

            adv_demand_charges = st.checkbox("Enable Demand Charges?",False)

            st.subheader("Seasonal Scale Factors")
            house_factors, solar_factors = [], []
            with st.expander("Adjust Monthly Factors"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf = st.slider(f"{mn} House Factor",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"housef_{i}")
                    sf = st.slider(f"{mn} Solar Factor",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"solarf_{i}")
                    house_factors.append(hf)
                    solar_factors.append(sf)

            st.subheader("Advanced EV")
            adv_ev_mean = st.slider("Mean Daily Miles (Adv EV)",0,100,30)
            adv_ev_std  = st.slider("StdDev Daily Miles",0,30,5)
            adv_ev_eff  = st.slider("EV Efficiency (miles/kWh)",3.0,5.0,4.0)
            adv_ev_cap  = st.slider("EV Battery Cap (kWh)",10,100,50)
            adv_ev_arrival = st.slider("EV Arrival Hour (AM/PM)",0,23,18)
            adv_ev_depart  = st.slider("EV Depart Hour",0,23,7)
            adv_home_batt  = st.slider("Home Battery (kWh, Adv LP)",0,40,10)

            st.write("**Running the LP** (365 daily solves)...")
            adv_daily_house, adv_daily_solar = build_daily_arrays_with_factors(
                house_kwh_base, solar_size, house_factors, solar_factors, fluct
            )
            adv_daily_ev = build_daily_ev_profile(adv_ev_mean, adv_ev_std, adv_ev_eff, adv_ev_cap)

            adv_cost, df_adv = run_advanced_lp_sim(
                adv_daily_house,
                adv_daily_solar,
                adv_daily_ev,
                ev_arrival_hr=adv_ev_arrival,
                ev_departure_hr=adv_ev_depart,
                home_batt_capacity=adv_home_batt,
                ev_battery_cap=adv_ev_cap,
                demand_charge_enabled=adv_demand_charges,
                battery_mode=adv_batt_mode,
                backup_reserve_frac=backup_reserve_slider,
                self_consumption_excess=self_consumption_excess_handling
            )
            if adv_cost is not None:
                st.success(f"Simulation done! Total Annual Net Cost: ${adv_cost:,.2f}")
                total_import = df_adv["grid_import"].sum()
                total_export = df_adv["grid_export"].sum()
                st.write(f"Grid Imports: {total_import:,.1f} kWh;  Solar Exports: {total_export:,.1f} kWh")

                day_pick = st.slider("Pick a day (0-364) for detail",0,364,0)
                df_dayp = df_adv[df_adv["day_idx"]==day_pick].copy()
                df_dayp["hour_label"] = df_dayp["hour"].apply(hour_to_ampm_label)

                st.write(f"### Day {day_pick} Hourly Dispatch")
                figX, axX = plt.subplots()
                axX.plot(df_dayp["hour_label"], df_dayp["batt_in"], label="Battery In", color="orange")
                axX.plot(df_dayp["hour_label"], df_dayp["batt_out"], label="Battery Out", color="green")
                axX.plot(df_dayp["hour_label"], df_dayp["ev_charge"], label="EV Charge", color="red")
                axX.set_xlabel("Hour")
                axX.set_ylabel("kWh")
                plt.xticks(rotation=45)
                axX.legend()
                st.pyplot(figX)

                st.write("""
                **Interpreting**:
                - **TOU Arbitrage**: Battery out only in on-peak (16–19).
                - **Self-Consumption**: 
                  - If "Curtail leftover," net export is 0 each hour.
                  - If "Export leftover," any leftover solar is exported once battery+loads are met.
                - **Backup Priority**: Battery never drops below the chosen reserve.
                - **None**: pure cost optimization with no special constraints.
                """)
            else:
                st.warning("LP returned no solution or PuLP not installed.")


if __name__=="__main__":
    main()
