import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Attempt PuLP import
try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# --------------------------------------------------------------------------------
#                          GLOBAL CONSTANTS & SETUP
# --------------------------------------------------------------------------------

DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10  # kWh
DEFAULT_BATTERY_EFFICIENCY = 0.9  # 90%
DEFAULT_SOLAR_SIZE = 7.5  # kW
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8  # kWh/day
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2  # 20%
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SUMMER_MONTHS = [5, 6, 7, 8]
WINTER_MONTHS = [0,1,2,3,4,9,10,11]

# Basic Hourly model
HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16, 21)),
    "off_peak_hours": list(range(7, 16)) + [21, 22],
    "super_off_peak_hours": list(range(0, 7)) + [23],
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90

# Advanced Hourly LP
DAYS_PER_YEAR = 365
DEFAULT_SOLAR_FACTORS = [0.6, 0.65, 0.75, 0.90, 1.0, 1.2, 1.3, 1.25, 1.0, 0.8, 0.65, 0.55]
DEFAULT_LOAD_FACTORS  = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.3, 1.1, 1.0, 1.0, 1.1]


# --------------------------------------------------------------------------------
#                       1. HELPER FUNCTIONS: MONTHLY
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    return [daily_value * d for d in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    daily_demand = miles / efficiency
    total_days = days_per_week * 52
    yearly = daily_demand * total_days
    monthly = calculate_monthly_values(daily_demand * (days_per_week / 7.0))
    return yearly, monthly

def calculate_solar_production(size_kw):
    yearly = size_kw * 4 * 365
    monthly = calculate_monthly_values(size_kw * 4)
    return yearly, monthly


def calculate_monthly_costs(ev_monthly, solar_monthly, house_monthly,
                            battery_capacity, time_of_charging):
    """
    Compare:
      1) No Solar
      2) NEM 2.0 (with dimensionally correct netting)
      3) NEM 3.0 + Battery (naive monthly)
    """
    ev_cost_nosolar, ev_cost_nem2, ev_cost_nem3 = [], [], []
    tot_nosolar, tot_nem2, tot_nem3 = [], [], []

    battery_state = 0.0

    for m in range(12):
        if m in SUMMER_MONTHS:
            on_peak = 0.45
            off_peak = 0.25
            super_off = 0.12
        else:
            on_peak = 0.35
            off_peak = 0.20
            super_off = 0.10

        # No Solar
        house_cost_ns = house_monthly[m]*off_peak
        ev_cost_ns    = ev_monthly[m]*super_off
        ev_cost_nosolar.append(ev_cost_ns)
        tot_nosolar.append(house_cost_ns + ev_cost_ns)

        # NEM 2.0
        # house cost still house_monthly[m]*off_peak
        house_cost_n2 = house_monthly[m]*off_peak
        leftover_solar_kwh = max(0, solar_monthly[m] - house_monthly[m])
        ev_kwh = ev_monthly[m]

        if time_of_charging=="Night (Super Off-Peak)":
            ev_rate = super_off
        else:
            ev_rate = on_peak

        if leftover_solar_kwh >= ev_kwh:
            # EV fully offset => ev cost = 0
            leftover_export_kwh = leftover_solar_kwh - ev_kwh
            ev_cost_2_val = 0.0
            # credit for leftover export
            credit_n2_val = leftover_export_kwh*off_peak
        else:
            # partial offset
            offset_kwh = leftover_solar_kwh
            leftover_export_kwh=0.0
            ev_grid_kwh = ev_kwh - offset_kwh
            ev_cost_2_val = ev_grid_kwh*ev_rate
            credit_n2_val = 0.0

        monthly_cost_n2 = house_cost_n2 + ev_cost_2_val - credit_n2_val
        ev_cost_nem2.append(ev_cost_2_val)
        tot_nem2.append(monthly_cost_n2)

        # NEM 3.0 + Battery (naive)
        cost_house_3 = house_monthly[m]*off_peak
        leftover_sol_3 = max(0, solar_monthly[m]-house_monthly[m])
        ev_short = ev_monthly[m]

        if time_of_charging=="Daytime (Peak)" and leftover_sol_3>0:
            direct_solar = min(ev_short, leftover_sol_3)
            ev_short -= direct_solar
            leftover_sol_3 -= direct_solar

        if leftover_sol_3>0 and battery_state<battery_capacity:
            can_charge = min(leftover_sol_3, battery_capacity-battery_state)
            battery_state += can_charge*DEFAULT_BATTERY_EFFICIENCY
            leftover_sol_3 -= can_charge

        if ev_short>0 and battery_state>0:
            discharge = min(ev_short, battery_state)
            ev_short -= discharge
            battery_state-=discharge

        if time_of_charging=="Night (Super Off-Peak)":
            ev_cost_3 = ev_short*super_off
        else:
            ev_cost_3 = ev_short*on_peak

        ev_cost_nem3.append(ev_cost_3)
        tot_nem3.append(cost_house_3+ev_cost_3)

    return (ev_cost_nosolar, ev_cost_nem2, ev_cost_nem3,
            tot_nosolar, tot_nem2, tot_nem3)


# --------------------------------------------------------------------------------
#       2. BASIC HOURLY MODEL
# --------------------------------------------------------------------------------

def classify_tou_basic(hour):
    if hour in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
    hour_of_day = hour_idx%24
    period = classify_tou_basic(hour_of_day)
    rate = HOUR_TOU_RATES_BASIC[period]

    total_demand = house_kwh+ev_kwh

    if solar_kwh>= total_demand:
        leftover_solar= solar_kwh-total_demand
        total_demand=0
    else:
        leftover_solar=0
        total_demand-=solar_kwh

    solar_unused=0
    if leftover_solar>0 and battery_state<battery_capacity:
        available_space = (battery_capacity-battery_state)/BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery = min(leftover_solar, available_space)
        battery_state+= to_battery*BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar-=to_battery
        solar_unused= leftover_solar
    else:
        solar_unused= leftover_solar

    if total_demand>0 and battery_state>0:
        discharge = min(total_demand,battery_state)
        total_demand-=discharge
        battery_state-=discharge

    grid_kwh= total_demand
    cost= grid_kwh*rate

    return battery_state, grid_kwh, cost, solar_unused

def run_basic_hourly_sim(daily_house, daily_solar, daily_ev,
                         battery_capacity=10.0,
                         ev_charging_pattern="Night",
                         reset_battery_daily=False):
    house_shape= np.array([
        0.02,0.02,0.02,0.02,0.02,0.03,0.04,0.06,
        0.06,0.06,0.06,0.05,0.05,0.05,0.06,0.07,
        0.08,0.06,0.05,0.05,0.04,0.03,0.02,0.02
    ])
    house_shape/= house_shape.sum()

    solar_shape= np.array([
        0.0,0.0,0.0,0.0,0.0,0.0,0.05,0.10,
        0.15,0.20,0.20,0.15,0.10,0.05,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    ])
    if solar_shape.sum()>0:
        solar_shape/= solar_shape.sum()

    if ev_charging_pattern=="Night":
        ev_shape = np.array([
            0.3,0.3,0.3,0.1,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ])
    else:
        ev_shape= np.array([
            0.0,0.0,0.0,0.0,0.0,0.05,0.10,0.15,
            0.15,0.15,0.15,0.10,0.05,0.0,0.0,0.0,
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ])
    if ev_shape.sum()>0:
        ev_shape/= ev_shape.sum()

    results= {
        "day":[],"hour":[],"house_kwh":[],"ev_kwh":[],"solar_kwh":[],"grid_kwh":[],
        "cost":[],"battery_state":[],"solar_unused":[]
    }

    days=len(daily_house)
    battery_state=0.0
    total_cost=0.0
    total_grid=0.0
    total_solar_unused=0.0

    for d in range(days):
        if reset_battery_daily:
            battery_state=0.0

        dh= daily_house[d]
        ds= daily_solar[d]
        de= daily_ev[d]

        # distribute daily => 24
        h_24 = dh*house_shape
        s_24 = ds*solar_shape
        e_24 = de*ev_shape

        for hour in range(24):
            hour_idx= d*24+hour
            bh= h_24[hour]
            bs= s_24[hour]
            be= e_24[hour]

            battery_state, g_kwh, cost, sol_un = simulate_hour_basic(
                hour_idx, bs, bh, be, battery_state, battery_capacity
            )
            total_cost+= cost
            total_grid+= g_kwh
            total_solar_unused+= sol_un

            results["day"].append(d)
            results["hour"].append(hour)
            results["house_kwh"].append(bh)
            results["ev_kwh"].append(be)
            results["solar_kwh"].append(bs)
            results["grid_kwh"].append(g_kwh)
            results["cost"].append(cost)
            results["battery_state"].append(battery_state)
            results["solar_unused"].append(sol_un)

    df= pd.DataFrame(results)
    return total_cost, total_grid, total_solar_unused, df

# --------------------------------------------------------------------------------
#               3. ADVANCED HOURLY LP (MODES)
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    data=[]
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    import_r = 0.45
                    export_r = 0.12
                    demand_r= 10.0 if d_type=="weekday" else 8.0
                elif 7<=h<16 or 20<=h<22:
                    import_r=0.25
                    export_r=0.08
                    demand_r=5.0
                else:
                    import_r=0.12
                    export_r=0.04
                    demand_r=2.0
                data.append({
                    "month":m,
                    "day_type":d_type,
                    "hour":h,
                    "import_rate": import_r,
                    "export_rate": export_r,
                    "demand_rate": demand_r
                })
    return pd.DataFrame(data)

def build_daily_arrays_with_factors(base_house_kwh, base_solar_kw,
                                    house_factors, solar_factors, fluct=0.0):
    daily_house=[]
    daily_solar=[]
    day_count=0
    for m, ndays in enumerate(DAYS_IN_MONTH):
        for _ in range(ndays):
            hv= base_house_kwh*house_factors[m]*(1+fluct)
            sv= (base_solar_kw*4)*solar_factors[m]
            daily_house.append(hv)
            daily_solar.append(sv)
            day_count+=1
            if day_count>=DAYS_PER_YEAR:
                break
        if day_count>=DAYS_PER_YEAR:
            break
    daily_house= np.array(daily_house[:DAYS_PER_YEAR])
    daily_solar= np.array(daily_solar[:DAYS_PER_YEAR])
    return daily_house, daily_solar

def build_daily_ev_profile(daily_miles_mean=30, daily_miles_std=5, ev_eff=4.0, ev_batt_cap=50):
    rng= np.random.default_rng(42)
    arr=[]
    for _ in range(DAYS_PER_YEAR):
        miles= rng.normal(daily_miles_mean, daily_miles_std)
        miles= max(0,miles)
        needed= miles/ev_eff
        needed= min(needed, ev_batt_cap)
        arr.append(needed)
    return np.array(arr)

def hour_to_ampm_label(h):
    if h==0: return "12 AM"
    elif h<12: return f"{h} AM"
    elif h==12:return "12 PM"
    else: return f"{h-12} PM"

def advanced_battery_constraints(prob, hour, hour_of_day,
                                 battery_mode, backup_reserve_frac,
                                 home_batt_out, grid_export, soc, batt_cap,
                                 self_consumption_excess="Curtail"):
    if battery_mode=="TOU Arbitrage":
        if hour_of_day<16 or hour_of_day>=20:
            prob += home_batt_out[hour]==0, f"TOU_no_out_{hour}"
    elif battery_mode=="Self-Consumption":
        if self_consumption_excess=="Curtail":
            prob += grid_export[hour]==0, f"NoExport_{hour}"
    elif battery_mode=="Backup Priority":
        prob += soc[hour]>= backup_reserve_frac*batt_cap, f"MinRes_{hour}"
    # "None" => no special constraints

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
    if not PULP_AVAILABLE:
        return None, start_batt_soc, pd.DataFrame()

    prob= LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc           = [pulp.LpVariable(f"soc_{h}",lowBound=0, upBound=home_batt_capacity) for h in range(25)]

    peak_demand= pulp.LpVariable("peak_demand",lowBound=0)

    cost_import=[]
    credit_export=[]

    for h in range(24):
        import_r= df_rates_day.loc[h,"import_rate"]
        export_r= df_rates_day.loc[h,"export_rate"]

        if not (ev_arrival<=h<ev_depart):
            prob += ev_charge[h]==0,f"EVNoCharge_{h}"

        prob += (
            solar_24[h] + home_batt_out[h] + grid_import[h]
            == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
        ), f"Balance_{h}"

        prob += soc[h+1]== soc[h] + home_batt_in[h] - home_batt_out[h], f"SOC_{h}"

        if demand_charge_enabled:
            prob += peak_demand>= grid_import[h], f"Peak_{h}"

        cost_import.append(grid_import[h]*import_r)
        credit_export.append(grid_export[h]*export_r)

        # battery constraints
        advanced_battery_constraints(prob,h,h,
                                     battery_mode,backup_reserve_frac,
                                     home_batt_out, grid_export, soc, home_batt_capacity,
                                     self_consumption_excess)

    # EV daily total
    prob += sum(ev_charge[h] for h in range(24))== ev_needed_kwh, "EVReq"

    prob += soc[0]== start_batt_soc,"StartSOC"

    total_import_cost= sum(cost_import)
    total_export_credit= sum(credit_export)
    demand_cost=0
    if demand_charge_enabled:
        max_dem_rate_day= df_rates_day["demand_rate"].max()
        demand_cost= peak_demand*max_dem_rate_day

    prob.setObjective(total_import_cost - total_export_credit + demand_cost)
    solver= pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status]=="Optimal":
        day_cost= pulp.value(prob.objective)
        end_soc= pulp.value(soc[24])
    else:
        day_cost=None
        end_soc= start_batt_soc

    hours_solution=[]
    for h in range(24):
        row= {
            "hour":h,
            "grid_import": pulp.value(grid_import[h]),
            "grid_export": pulp.value(grid_export[h]),
            "batt_in":     pulp.value(home_batt_in[h]),
            "batt_out":    pulp.value(home_batt_out[h]),
            "ev_charge":   pulp.value(ev_charge[h]),
        }
        hours_solution.append(row)
    df_day= pd.DataFrame(hours_solution)
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
    if not PULP_AVAILABLE:
        return None, pd.DataFrame()

    df_rates= generate_utility_rate_schedule()
    cum_days= np.cumsum([0]+DAYS_IN_MONTH)
    day_solutions=[]
    total_cost=0.0
    battery_soc=0.0

    for day_idx in range(DAYS_PER_YEAR):
        m=0
        while m<12 and day_idx>=cum_days[m+1]:
            m+=1
        dow= day_idx%7
        d_type= "weekend" if dow in [5,6] else "weekday"

        df_day_rates= df_rates[(df_rates["month"]==m)&(df_rates["day_type"]==d_type)].copy()
        df_day_rates.sort_values("hour",inplace=True)
        df_day_rates.set_index("hour",inplace=True)

        dh= daily_house[day_idx]
        ds= daily_solar[day_idx]
        de= daily_ev[day_idx]

        house_24= np.full(24, dh/24.0)
        solar_24= np.full(24, ds/24.0)

        day_cost, end_soc, df_day_sol = optimize_daily(
            day_idx, house_24, solar_24, de,
            ev_arrival_hr, ev_departure_hr,
            battery_soc, home_batt_capacity,
            ev_battery_cap, df_day_rates,
            demand_charge_enabled,
            battery_mode,
            backup_reserve_frac,
            self_consumption_excess
        )
        if day_cost is not None:
            total_cost+= day_cost
        battery_soc= end_soc

        df_day_sol["day_idx"]= day_idx
        df_day_sol["month"]= m
        df_day_sol["day_type"]= d_type
        df_day_sol["cost_day"]= day_cost if day_cost else 0
        day_solutions.append(df_day_sol)

    df_sol= pd.concat(day_solutions, ignore_index=True)
    return total_cost, df_sol


# --------------------------------------------------------------------------------
#                         STREAMLIT MAIN APP
# --------------------------------------------------------------------------------

def main():
    st.title("Unified App with Expanded Charts & Explanations")

    st.write("""
    This app demonstrates:
    1. A **Monthly Approach** with **fixed** NEM 2.0 dimension mismatch (using kWh netting).
    2. A **Basic Hourly** approach (naive battery, no net export credit).
    3. An **Advanced Hourly LP** with multiple discharge modes (TOU Arbitrage, 
       Self-Consumption, Backup Priority, None), plus an option to curtail or export leftover solar 
       in Self-Consumption mode.
    4. **Multiple charts** in each tab to help interpret the results.
    """)

    st.sidebar.header("Common Inputs")

    commute_miles = st.sidebar.slider("Daily Commute (miles)",10,100, DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_freq = st.sidebar.radio("EV Charging Frequency (Monthly & Basic Hourly)", ["Daily","Weekdays Only"])
    days_per_week = 5 if charging_freq=="Weekdays Only" else 7

    monthly_charge_time = st.sidebar.radio("Monthly Model: EV Charging Time", ["Night (Super Off-Peak)","Daytime (Peak)"])

    # House
    house_kwh_base = st.sidebar.slider("Daily House (kWh)",10,50,int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluct = st.sidebar.slider("House Fluctuation (%)",0,50,int(DEFAULT_CONSUMPTION_FLUCTUATION*100))/100

    # Solar & Battery (Monthly/Basic)
    solar_size = st.sidebar.slider("Solar Size (kW)",0,15,int(DEFAULT_SOLAR_SIZE))
    monthly_batt_capacity = st.sidebar.slider("Battery (kWh) for Monthly/Basic Hourly",0,20,int(DEFAULT_BATTERY_CAPACITY))

    tab1, tab2, tab3 = st.tabs(["Monthly Approach","Basic Hourly","Advanced Hourly LP (Modes)"])

    # ---------------------------------------------
    # TAB 1: MONTHLY
    # ---------------------------------------------
    with tab1:
        st.header("Monthly Net Approach")

        # EV demand monthly
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, efficiency, days_per_week)
        daily_house_val = house_kwh_base*(1+fluct)
        house_monthly= calculate_monthly_values(daily_house_val)
        _, solar_monthly= calculate_solar_production(solar_size)

        (ev_nosolar, ev_n2, ev_n3,
         tot_nosolar, tot_n2, tot_n3) = calculate_monthly_costs(
            ev_monthly, solar_monthly, house_monthly,
            monthly_batt_capacity, monthly_charge_time
        )

        df_monthly = pd.DataFrame({
            "Month": MONTH_NAMES,
            "House (kWh)": house_monthly,
            "EV (kWh)": ev_monthly,
            "Solar (kWh)": solar_monthly,
            "EV_Cost_NoSolar($)": ev_nosolar,
            "EV_Cost_NEM2($)": ev_n2,
            "EV_Cost_NEM3($)": ev_n3,
            "Total_NoSolar($)": tot_nosolar,
            "Total_NEM2($)": tot_n2,
            "Total_NEM3($)": tot_n3
        })

        # Display table
        st.subheader("Monthly Results Table")
        st.dataframe(df_monthly.style.format(precision=2))

        # Summaries
        st.write("### Annual Summaries")
        st.write(f"**Annual EV kWh**: {sum(ev_monthly):.1f}")
        st.write(f"**Annual House kWh**: {sum(house_monthly):.1f}")
        st.write(f"**Annual Solar**: {sum(solar_monthly):.1f} kWh")

        cost_no_solar_annual= sum(tot_nosolar)
        cost_nem2_annual= sum(tot_n2)
        cost_nem3_annual= sum(tot_n3)
        st.write(f"**Total Cost (No Solar)**: ${cost_no_solar_annual:.2f}")
        st.write(f"**Total Cost (NEM 2.0)**: ${cost_nem2_annual:.2f}")
        st.write(f"**Total Cost (NEM 3.0 + Batt)**: ${cost_nem3_annual:.2f}")

        # Some charts
        st.subheader("Charts")

        # 1) Bar chart: House vs. EV vs. Solar
        fig1, ax1 = plt.subplots()
        width=0.25
        x= np.arange(12)
        ax1.bar(x - width, df_monthly["House (kWh)"], width, label="House")
        ax1.bar(x, df_monthly["EV (kWh)"], width, label="EV")
        ax1.bar(x + width, df_monthly["Solar (kWh)"], width, label="Solar")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_monthly["Month"])
        ax1.set_ylabel("kWh")
        ax1.set_title("Monthly House, EV, and Solar")
        ax1.legend()
        st.pyplot(fig1)

        st.write("""
        *Interpretation*: 
        - This bar chart compares monthly *kWh* for house load, EV usage, and solar generation. 
        - Notice if solar exceeds (House+EV) or not each month.
        """)

        # 2) EV Charging Cost lines
        fig2, ax2= plt.subplots()
        ax2.plot(df_monthly["Month"], df_monthly["EV_Cost_NoSolar($)"], label="EV (No Solar)", marker="o")
        ax2.plot(df_monthly["Month"], df_monthly["EV_Cost_NEM2($)"], label="EV (NEM2)", marker="s")
        ax2.plot(df_monthly["Month"], df_monthly["EV_Cost_NEM3($)"], label="EV (NEM3)", marker="^")
        ax2.set_ylabel("EV Cost ($)")
        ax2.set_title("Monthly EV Charging Cost")
        ax2.legend()
        st.pyplot(fig2)

        st.write("""
        *Interpretation*:
        - Under NEM 2.0, EV cost can drop if leftover solar offsets or is credited at near-retail. 
        - "No Solar" is simply EV kWh * super-off-peak rate for each month.
        - NEM 3.0 here is a naive monthly approach with a battery, might not show huge savings 
          if the monthly logic doesn't fully capture daily TOU differences.
        """)

        # 3) Total monthly cost lines
        fig3, ax3= plt.subplots()
        ax3.plot(df_monthly["Month"], df_monthly["Total_NoSolar($)"], label="No Solar", marker="o")
        ax3.plot(df_monthly["Month"], df_monthly["Total_NEM2($)"], label="NEM 2.0", marker="s")
        ax3.plot(df_monthly["Month"], df_monthly["Total_NEM3($)"], label="NEM 3.0+Batt", marker="^")
        ax3.set_ylabel("Monthly Cost ($)")
        ax3.set_title("Total Monthly Cost Comparison")
        ax3.legend()
        st.pyplot(fig3)

        st.write("""
        *Interpretation*:
        - Total cost includes House + EV minus any export credits. 
        - If NEM 2.0 is lower overall, it means the monthly net logic is allowing a decent credit 
          for leftover solar vs. the EV usage or exports.
        """)


    # ---------------------------------------------
    # TAB 2: BASIC HOURLY
    # ---------------------------------------------
    with tab2:
        st.header("Basic Hourly Approach")

        daily_house_arr = np.full(DAYS_PER_YEAR, house_kwh_base*(1+fluct))
        daily_solar_arr = np.full(DAYS_PER_YEAR, solar_size*4)
        daily_ev_arr    = np.full(DAYS_PER_YEAR, (commute_miles/efficiency)*(days_per_week/7.0))

        reset_daily_batt= st.checkbox("Reset Battery Daily? (Basic Hourly)",False)
        ev_basic_pattern= st.selectbox("EV Charging Pattern (Night/Daytime)",["Night","Daytime"])

        cost_basic, grid_basic, solar_un_basic, df_basic= run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            monthly_batt_capacity,
            ev_charging_pattern= ev_basic_pattern,
            reset_battery_daily= reset_daily_batt
        )

        st.write(f"**Total Annual Cost**: ${cost_basic:,.2f}")
        st.write(f"**Total Grid Usage**: {grid_basic:,.0f} kWh")
        st.write(f"**Unused Solar**: {solar_un_basic:,.0f} kWh")

        # Possibly show a daily cost or something
        # compute daily cost from df_basic
        daily_costs_basic= df_basic.groupby("day")["cost"].sum().reset_index(name="day_cost")
        # Plot daily cost distribution
        st.subheader("Daily Cost Over the Year")
        figA, axA= plt.subplots()
        axA.plot(daily_costs_basic["day"], daily_costs_basic["day_cost"], label="Daily Cost ($)")
        axA.set_xlabel("Day of Year")
        axA.set_ylabel("Cost ($)")
        axA.set_title("Daily Cost (Basic Hourly)")
        axA.legend()
        st.pyplot(figA)

        st.write("""
        *Interpretation*:
        - This line shows how the daily cost can vary over the year, depending on solar production 
          and EV usage on each day. 
        - If solar is large in summer, costs might drop then, etc.
        """)

        # Let user pick a day to see hourly breakdown
        day_sel= st.slider("Pick a day (0-364) for Hourly Graphs",0,364,0)
        df_day_sel= df_basic[df_basic["day"]== day_sel].copy()
        df_day_sel["hour_label"]= df_day_sel["hour"].apply(hour_to_ampm_label)

        st.subheader(f"Hourly Breakdown - Day {day_sel}")
        figB, axB= plt.subplots()
        axB.plot(df_day_sel["hour_label"], df_day_sel["house_kwh"], label="House")
        axB.plot(df_day_sel["hour_label"], df_day_sel["ev_kwh"], label="EV", linestyle="--")
        axB.plot(df_day_sel["hour_label"], df_day_sel["solar_kwh"], label="Solar", color="gold")
        plt.xticks(rotation=45)
        axB.set_ylabel("kWh")
        axB.set_title(f"Day {day_sel} - House, EV, Solar")
        axB.legend()
        st.pyplot(figB)

        st.write("""
        *Interpretation*:
        - This hourly chart shows how solar production (kWh) compares to the house load and EV load each hour.
        - If solar is bigger than total load, leftover is "unused" since there's no net export credit in the basic model.
        """)

        figB2, axB2= plt.subplots()
        axB2.plot(df_day_sel["hour_label"], df_day_sel["battery_state"], color="green", label="Battery State (kWh)")
        axB2.set_ylabel("Battery (kWh)")
        axB2.set_xlabel("Hour")
        plt.xticks(rotation=45)
        axB2.legend(loc="upper left")

        axB3= axB2.twinx()
        axB3.plot(df_day_sel["hour_label"], df_day_sel["grid_kwh"], color="red", label="Grid Import (kWh)")
        axB3.set_ylabel("Grid (kWh)")
        axB3.legend(loc="upper right")
        st.pyplot(figB2)

        st.write("""
        *Interpretation*: 
        - We see how battery charges/discharges each hour, and how much is pulled from the grid.
        - If battery or solar is insufficient, more grid usage is required.
        """)


    # ---------------------------------------------
    # TAB 3: ADVANCED HOURLY LP (MODES)
    # ---------------------------------------------
    with tab3:
        st.header("Advanced Hourly LP with Multiple Discharge Modes & Excess Solar Toggle")

        if not PULP_AVAILABLE:
            st.error("PuLP not installed. Please install for this tab.")
        else:
            st.write("""
            **Modes**:
            - **None**: unconstrained cost optimization.
            - **TOU Arbitrage**: discharge only on-peak (16-19).
            - **Self-Consumption**: leftover solar can be 'Curtail' (no export) or 'Export'.
            - **Backup Priority**: keep battery >= some fraction. 
            """)

            adv_batt_mode= st.selectbox("Battery Mode",["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_reserve=0.0
            self_consumption_excess="Curtail"
            if adv_batt_mode=="Backup Priority":
                backup_reserve= st.slider("Backup Reserve (%)",0,50,20)/100
            elif adv_batt_mode=="Self-Consumption":
                self_consumption_excess= st.radio("Excess Solar in Self-Consumption?",["Curtail","Export"])

            adv_demand_charges= st.checkbox("Enable Demand Charges?",False)

            st.subheader("Seasonal Scale Factors for House & Solar")
            house_factors=[]
            solar_factors=[]
            with st.expander("Adjust Monthly Factors"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf= st.slider(f"{mn} House Factor",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"adv_housef_{i}")
                    sf= st.slider(f"{mn} Solar Factor",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"adv_solarf_{i}")
                    house_factors.append(hf)
                    solar_factors.append(sf)

            st.subheader("Advanced EV Settings")
            adv_ev_mean= st.slider("Mean Daily Miles (Adv EV)",0,100,30)
            adv_ev_std= st.slider("StdDev Daily Miles",0,30,5)
            adv_ev_eff= st.slider("EV Efficiency (miles/kWh)",3.0,5.0,4.0)
            adv_ev_cap= st.slider("EV Battery Cap (kWh)",10,100,50)
            adv_ev_arrival= st.slider("EV Arrival Hour",0,23,18)
            adv_ev_depart= st.slider("EV Depart Hour",0,23,7)
            adv_home_batt= st.slider("Home Battery (kWh, Adv LP)",0,40,10)

            st.write("**Running the LP** for 365 days...")

            # Build arrays
            adv_daily_house, adv_daily_solar = build_daily_arrays_with_factors(
                house_kwh_base, solar_size, house_factors, solar_factors, fluct
            )
            adv_daily_ev = build_daily_ev_profile(adv_ev_mean, adv_ev_std, adv_ev_eff, adv_ev_cap)

            adv_cost, df_adv_sol = run_advanced_lp_sim(
                adv_daily_house,
                adv_daily_solar,
                adv_daily_ev,
                ev_arrival_hr= adv_ev_arrival,
                ev_departure_hr= adv_ev_depart,
                home_batt_capacity= adv_home_batt,
                ev_battery_cap= adv_ev_cap,
                demand_charge_enabled= adv_demand_charges,
                battery_mode= adv_batt_mode,
                backup_reserve_frac= backup_reserve,
                self_consumption_excess=self_consumption_excess
            )
            if adv_cost is not None:
                st.success(f"Done! Total Annual Net Cost: ${adv_cost:,.2f}")
                total_import= df_adv_sol["grid_import"].sum()
                total_export= df_adv_sol["grid_export"].sum()
                st.write(f"**Grid Imports**: {total_import:,.1f} kWh,  **Solar Exports**: {total_export:,.1f} kWh")

                # Plot daily cost distribution
                daily_costs_lp= df_adv_sol.groupby("day_idx")["cost_day"].first().reset_index()
                # or sum if day has partial cost, but "cost_day" is repeated each row => you might do .first() or .mean()
                st.subheader("Daily Cost Over the Year (LP)")
                figC, axC= plt.subplots()
                axC.plot(daily_costs_lp["day_idx"], daily_costs_lp["cost_day"], label="Daily Cost ($)")
                axC.set_xlabel("Day of Year")
                axC.set_ylabel("Cost ($)")
                axC.set_title("Daily Cost from the Advanced LP")
                axC.legend()
                st.pyplot(figC)

                st.write("""
                *Interpretation*:
                - This line shows how cost varies day to day with advanced dispatch, 
                  different battery usage, EV patterns, etc.
                """)

                # Let user pick a day
                day_pick= st.slider("Pick a day (0-364) for Hourly Dispatch",0,364,0)
                df_dayp= df_adv_sol[df_adv_sol["day_idx"]== day_pick].copy()
                df_dayp["hour_label"]= df_dayp["hour"].apply(hour_to_ampm_label)

                st.subheader(f"Day {day_pick} Hourly Dispatch")
                figD, axD= plt.subplots()
                axD.plot(df_dayp["hour_label"], df_dayp["batt_in"], label="Battery In", color="orange")
                axD.plot(df_dayp["hour_label"], df_dayp["batt_out"], label="Battery Out", color="green")
                axD.plot(df_dayp["hour_label"], df_dayp["ev_charge"], label="EV Charge", color="red")
                axD.set_xlabel("Hour")
                axD.set_ylabel("kWh")
                plt.xticks(rotation=45)
                axD.legend()
                st.pyplot(figD)

                st.write("""
                *Interpretation*:
                - This hourly chart shows how the LP is scheduling battery 
                  charging/discharging and EV charging each hour.
                - If "Self-Consumption: Curtail," you may see no grid_export at all. 
                  If "Export," leftover solar might appear as grid_export > 0 in some hours.
                """)
            else:
                st.warning("No solution or PuLP missing.")


if __name__=="__main__":
    main()
