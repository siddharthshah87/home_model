import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Attempt PuLP
try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# --------------------------------------------------------------------------------
#                           GLOBALS & CONSTANTS
# --------------------------------------------------------------------------------

DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10
DEFAULT_BATTERY_EFFICIENCY = 0.9
DEFAULT_SOLAR_SIZE = 7.5
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2

DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SUMMER_MONTHS = [5,6,7,8]
WINTER_MONTHS = [0,1,2,3,4,9,10,11]

DAYS_PER_YEAR = 365

# Basic Hourly
HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16,21)),
    "off_peak_hours": list(range(7,16)) + [21,22],
    "super_off_peak_hours": list(range(0,7)) + [23],
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90

# Advanced approach placeholders
DEFAULT_SOLAR_FACTORS = [0.6,0.65,0.75,0.90,1.0,1.2,1.3,1.25,1.0,0.8,0.65,0.55]
DEFAULT_LOAD_FACTORS = [1.1,1.0,0.9,0.9,1.0,1.2,1.3,1.3,1.1,1.0,1.0,1.1]

# --------------------------------------------------------------------------------
#                          1. MONTHLY MODEL
# --------------------------------------------------------------------------------

def calculate_monthly_values(daily_value):
    return [daily_value*d for d in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    daily_demand = miles/ efficiency
    total_days = days_per_week*52
    yearly= daily_demand* total_days
    monthly= calculate_monthly_values(daily_demand*(days_per_week/7.0))
    return yearly, monthly

def calculate_solar_production(size_kw):
    yearly= size_kw*4*365
    monthly= calculate_monthly_values(size_kw*4)
    return yearly, monthly

def calculate_monthly_costs(ev_monthly, solar_monthly, house_monthly,
                            battery_capacity, time_of_charging):
    # Similar logic as before (1) No Solar, (2) NEM2, (3) NEM3 naive
    # We'll do a simplified approach
    ev_ns=[]
    ev_n2=[]
    ev_n3=[]
    tot_ns=[]
    tot_n2=[]
    tot_n3=[]
    battery_state=0.0

    for m in range(12):
        if m in SUMMER_MONTHS:
            on_peak=0.45
            off_peak=0.25
            super_off=0.12
        else:
            on_peak=0.35
            off_peak=0.20
            super_off=0.10

        # No solar
        cost_house_ns= house_monthly[m]* off_peak
        cost_ev_ns= ev_monthly[m]* super_off
        ev_ns.append(cost_ev_ns)
        tot_ns.append(cost_house_ns+ cost_ev_ns)

        # NEM2
        house_cost_n2= house_monthly[m]* off_peak
        leftover_solar_kwh= max(0, solar_monthly[m]- house_monthly[m])
        ev_kwh= ev_monthly[m]

        if time_of_charging=="Night (Super Off-Peak)":
            ev_rate= super_off
        else:
            ev_rate= on_peak

        if leftover_solar_kwh>= ev_kwh:
            leftover_export= leftover_solar_kwh- ev_kwh
            ev_cost2=0.0
            credit_n2= leftover_export* off_peak
        else:
            offset_kwh= leftover_solar_kwh
            leftover_export=0
            ev_grid_kwh= ev_kwh- offset_kwh
            ev_cost2= ev_grid_kwh* ev_rate
            credit_n2=0.0

        cost_n2= house_cost_n2+ ev_cost2 - credit_n2
        ev_n2.append(ev_cost2)
        tot_n2.append(cost_n2)

        # NEM3 + naive battery
        house_cost_3= house_monthly[m]* off_peak
        leftover_sol_3= max(0, solar_monthly[m]- house_monthly[m])
        ev_short= ev_monthly[m]

        if time_of_charging=="Daytime (Peak)" and leftover_sol_3>0:
            direct_solar= min(ev_short, leftover_sol_3)
            ev_short-= direct_solar
            leftover_sol_3-= direct_solar

        if leftover_sol_3>0 and battery_state< battery_capacity:
            can_chg= min(leftover_sol_3, battery_capacity- battery_state)
            battery_state+= can_chg* DEFAULT_BATTERY_EFFICIENCY
            leftover_sol_3-= can_chg

        if ev_short>0 and battery_state>0:
            discharge= min(ev_short, battery_state)
            ev_short-= discharge
            battery_state-= discharge

        if time_of_charging=="Night (Super Off-Peak)":
            ev_cost_3= ev_short* super_off
        else:
            ev_cost_3= ev_short* on_peak

        ev_n3.append(ev_cost_3)
        tot_n3.append(house_cost_3+ ev_cost_3)

    return (ev_ns, ev_n2, ev_n3,
            tot_ns, tot_n2, tot_n3)

# --------------------------------------------------------------------------------
#     2. BASIC HOURLY with Night wrap-around
# --------------------------------------------------------------------------------

def classify_tou_basic(hour):
    if hour in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh,
                        battery_state, battery_capacity):
    hour_of_day= hour_idx%24
    period= classify_tou_basic(hour_of_day)
    rate= HOUR_TOU_RATES_BASIC[period]

    total_demand= house_kwh+ ev_kwh
    if solar_kwh>= total_demand:
        leftover_solar= solar_kwh- total_demand
        total_demand=0
    else:
        leftover_solar=0
        total_demand-= solar_kwh

    solar_unused=0
    if leftover_solar>0 and battery_state< battery_capacity:
        space= (battery_capacity- battery_state)/ BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery= min(leftover_solar, space)
        battery_state+= to_battery* BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar-= to_battery
        solar_unused= leftover_solar
    else:
        solar_unused= leftover_solar

    if total_demand>0 and battery_state>0:
        discharge= min(total_demand, battery_state)
        total_demand-= discharge
        battery_state-= discharge

    grid_kwh= total_demand
    cost= grid_kwh* rate
    return battery_state, grid_kwh, cost, solar_unused

def build_basic_ev_shape(pattern="Night"):
    shape= np.zeros(24)
    if pattern=="Night":
        # 18..23 => 6 hours, 0..6 => 7 hours => total 13
        hours_night= list(range(18,24)) + list(range(0,7))
        for h in hours_night:
            shape[h]=1.0
        shape/= shape.sum()
    else:
        # "Daytime" => 9..16 => let's do 8 hours for example
        hours_day= list(range(9,17)) #9..16
        for h in hours_day:
            shape[h]= 1.0
        shape/= shape.sum()
    return shape

def run_basic_hourly_sim(
    daily_house,
    daily_solar,
    daily_ev,
    battery_capacity=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
    ev_shape= build_basic_ev_shape(ev_charging_pattern)

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

    results={
        "day":[],"hour":[],"house_kwh":[],"ev_kwh":[],"solar_kwh":[],
        "grid_kwh":[],"cost":[],"battery_state":[],"solar_unused":[]
    }

    battery_state=0.0
    total_cost=0.0
    total_grid=0.0
    total_solar_unused=0.0

    days= len(daily_house)
    for d in range(days):
        if reset_battery_daily:
            battery_state=0.0

        dh= daily_house[d]
        ds= daily_solar[d]
        de= daily_ev[d]

        h_24= dh* house_shape
        s_24= ds* solar_shape
        e_24= de* ev_shape

        for hour in range(24):
            hour_idx= d*24+ hour
            bh= h_24[hour]
            bs= s_24[hour]
            be= e_24[hour]

            battery_state, g_kwh, cost, sol_un= simulate_hour_basic(
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
#    3. ADVANCED HOURLY LP
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    """
    We'll define a single-season approach with a big gap 
    between import and export for no confusion
    """
    data=[]
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    import_r= 0.45
                    export_r= 0.05
                    demand_r= 10.0
                elif 7<=h<16 or 20<=h<22:
                    import_r= 0.25
                    export_r= 0.03
                    demand_r= 5.0
                else:
                    import_r= 0.12
                    export_r= 0.01
                    demand_r= 2.0
                data.append({
                    "month": m,
                    "day_type": d_type,
                    "hour": h,
                    "import_rate": import_r,
                    "export_rate": export_r,
                    "demand_rate": demand_r
                })
    return pd.DataFrame(data)

def can_charge_this_hour(h, arr, dep):
    if arr < dep:
        return arr <= h < dep
    else:
        return (h >= arr) or (h < dep)

def advanced_battery_constraints(prob, hour, hour_of_day,
                                 battery_mode, backup_reserve_frac,
                                 home_batt_out, grid_export, soc,
                                 batt_cap,
                                 self_consumption_excess="Curtail"):
    if battery_mode=="TOU Arbitrage":
        if hour_of_day<16 or hour_of_day>=20:
            prob += home_batt_out[hour]==0
    elif battery_mode=="Self-Consumption":
        if self_consumption_excess=="Curtail":
            prob += grid_export[hour]==0
    elif battery_mode=="Backup Priority":
        prob += soc[hour]>= backup_reserve_frac*batt_cap

def optimize_daily_lp(
    day_idx,
    house_24,
    solar_24,
    ev_kwh_day,
    ev_arr, ev_dep,
    start_batt_soc,
    battery_capacity,
    ev_batt_capacity,
    df_rates_day,
    demand_charge_enabled=False,
    battery_mode="None",
    backup_reserve_frac=0.2,
    self_consumption_excess="Curtail"
):
    # If battery_mode= None => battery=0
    if battery_mode=="None":
        battery_capacity=0

    prob= LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc           = [pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=battery_capacity) for h in range(25)]

    peak_demand= pulp.LpVariable("peak_demand", lowBound=0)
    cost_import=[]
    credit_export=[]

    for h in range(24):
        import_r= df_rates_day.loc[h,"import_rate"]
        export_r= df_rates_day.loc[h,"export_rate"]

        if not can_charge_this_hour(h, ev_arr, ev_dep):
            prob += ev_charge[h]==0

        # power balance
        prob += (
            solar_24[h] + home_batt_out[h] + grid_import[h]
            == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
        )

        # battery soc
        prob += soc[h+1]== soc[h] + home_batt_in[h] - home_batt_out[h]

        if demand_charge_enabled:
            prob += peak_demand>= grid_import[h]

        cost_import.append(grid_import[h]* import_r)
        credit_export.append(grid_export[h]* export_r)

        advanced_battery_constraints(prob,h,h,battery_mode,backup_reserve_frac,
                                     home_batt_out, grid_export, soc,
                                     battery_capacity,
                                     self_consumption_excess)

    prob += sum(ev_charge[h] for h in range(24))== ev_kwh_day

    # day reset
    prob += soc[0]== start_batt_soc
    prob += soc[24]== soc[0]

    # enforce soc <= capacity
    for hh in range(25):
        prob += soc[hh] <= battery_capacity

    total_import_cost= sum(cost_import)
    total_export_credit= sum(credit_export)
    demand_cost=0
    if demand_charge_enabled:
        max_dem= df_rates_day["demand_rate"].max()
        demand_cost= peak_demand* max_dem

    prob.setObjective(total_import_cost - total_export_credit + demand_cost)
    solver= pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status]=="Optimal":
        day_cost= pulp.value(prob.objective)
        end_soc= pulp.value(soc[24])
    else:
        day_cost= None
        end_soc= start_batt_soc

    return day_cost, end_soc

def run_advanced_lp_sim(
    daily_house,
    daily_solar,
    daily_ev,
    ev_arr=18,
    ev_dep=7,
    battery_capacity=10,
    ev_batt_capacity=50,
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
        while m<12 and day_idx>= cum_days[m+1]:
            m+=1
        dow= day_idx%7
        d_type= "weekend" if dow in [5,6] else "weekday"

        df_day= df_rates[(df_rates["month"]==m)&(df_rates["day_type"]==d_type)].copy()
        df_day.sort_values("hour", inplace=True)
        df_day.set_index("hour", inplace=True)

        dh= daily_house[day_idx]
        ds= daily_solar[day_idx]
        de= daily_ev[day_idx]

        house_24= np.full(24, dh/24.0)
        solar_24= np.full(24, ds/24.0)

        day_cost, end_soc= optimize_daily_lp(
            day_idx, house_24, solar_24, de,
            ev_arr, ev_dep,
            battery_soc, battery_capacity, ev_batt_capacity,
            df_day,
            demand_charge_enabled,
            battery_mode,
            backup_reserve_frac,
            self_consumption_excess
        )
        if day_cost is not None:
            total_cost+= day_cost
            battery_soc= end_soc

        day_solutions.append({"day_idx":day_idx, "cost_day": day_cost if day_cost else 0})

    df_sol= pd.DataFrame(day_solutions)
    return total_cost, df_sol

# --------------------------------------------------------------------------------
#  5. URDB Key from secrets + main app
# --------------------------------------------------------------------------------

def main():
    st.title("App Using URDB Key from secrets.toml")

    st.write("""
    We are pulling `URDB_API_KEY` from `.streamlit/secrets.toml`.
    Example of secrets.toml:
    ```
    URDB_API_KEY="your_key_here"
    ```
    """)

    # We retrieve the key from st.secrets
    api_key = st.secrets.get("URDB_API_KEY", "")
    st.write("**URDB API Key** (from secrets.toml):", 
             f"{api_key[:5]}... (hidden rest)" if api_key else "(None found)")

    st.sidebar.header("Common Inputs")

    # ... the rest of your code for user inputs, 
    # battery slider, monthly approach, basic hourly, advanced LP, etc...
    # This code is the same as the prior example. 
    # We'll not re-duplicate everything for brevity. 
    # Just to show how we store and read the URDB key from secrets.
    
    st.write("Rest of the app using the single battery slider, advanced approach, etc...")

if __name__=="__main__":
    main()
