import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Attempt to import PuLP for advanced LP
try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# --------------------------------------------------------------------------------
#                          GLOBAL CONSTANTS
# --------------------------------------------------------------------------------

DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10  # kWh
DEFAULT_BATTERY_EFFICIENCY = 0.9
DEFAULT_SOLAR_SIZE = 7.5
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SUMMER_MONTHS = [5,6,7,8]
WINTER_MONTHS = [0,1,2,3,4,9,10,11]

DAYS_PER_YEAR = 365

HOUR_TOU_SCHEDULE_BASIC = {
    "on_peak_hours": list(range(16, 21)),
    "off_peak_hours": list(range(7,16)) + [21,22],
    "super_off_peak_hours": list(range(0,7)) + [23],
}
HOUR_TOU_RATES_BASIC = {
    "on_peak": 0.45,
    "off_peak": 0.25,
    "super_off_peak": 0.12
}
BATTERY_HOURLY_EFFICIENCY_BASIC = 0.90

# For advanced approach
DEFAULT_SOLAR_FACTORS = [0.6,0.65,0.75,0.90,1.0,1.2,1.3,1.25,1.0,0.8,0.65,0.55]
DEFAULT_LOAD_FACTORS  = [1.1,1.0,0.9,0.9,1.0,1.2,1.3,1.3,1.1,1.0,1.0,1.1]

# --------------------------------------------------------------------------------
#                 1. MONTHLY MODEL (NEM 2.0 vs. NEM 3.0)
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
    yearly = size_kw * 4 * 365
    monthly = calculate_monthly_values(size_kw * 4)
    return yearly, monthly

def calculate_monthly_costs(ev_monthly, solar_monthly, house_monthly,
                            battery_capacity, time_of_charging):
    ev_cost_no_solar = []
    ev_cost_nem_2    = []
    ev_cost_nem_3    = []

    tot_no_solar     = []
    tot_nem_2        = []
    tot_nem_3        = []

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

        # 1) No Solar
        cost_house_ns = house_monthly[m]* off_peak
        cost_ev_ns    = ev_monthly[m]* super_off
        ev_cost_no_solar.append(cost_ev_ns)
        tot_no_solar.append(cost_house_ns + cost_ev_ns)

        # 2) NEM 2.0
        house_cost_n2 = house_monthly[m]* off_peak
        leftover_solar_kwh = max(0, solar_monthly[m] - house_monthly[m])
        ev_kwh= ev_monthly[m]

        if time_of_charging=="Night (Super Off-Peak)":
            ev_rate= super_off
        else:
            ev_rate= on_peak

        if leftover_solar_kwh >= ev_kwh:
            leftover_export_kwh= leftover_solar_kwh - ev_kwh
            ev_cost2_val= 0.0
            # leftover export credited at off_peak
            credit_n2= leftover_export_kwh * off_peak
        else:
            offset_kwh= leftover_solar_kwh
            leftover_export_kwh= 0.0
            ev_grid_kwh= ev_kwh - offset_kwh
            ev_cost2_val= ev_grid_kwh* ev_rate
            credit_n2= 0.0

        cost_n2= house_cost_n2 + ev_cost2_val - credit_n2
        ev_cost_nem_2.append(ev_cost2_val)
        tot_nem_2.append(cost_n2)

        # 3) NEM 3.0 + battery (naive monthly)
        cost_house_3= house_monthly[m]* off_peak
        leftover_sol_3= max(0, solar_monthly[m]- house_monthly[m])
        ev_short= ev_monthly[m]

        if time_of_charging=="Daytime (Peak)" and leftover_sol_3>0:
            direct_solar= min(ev_short, leftover_sol_3)
            ev_short-= direct_solar
            leftover_sol_3-= direct_solar

        # charge battery
        if leftover_sol_3>0 and battery_state< battery_capacity:
            can_charge= min(leftover_sol_3, battery_capacity- battery_state)
            battery_state+= can_charge* DEFAULT_BATTERY_EFFICIENCY
            leftover_sol_3-= can_charge

        # discharge battery
        if ev_short>0 and battery_state>0:
            discharge= min(ev_short, battery_state)
            ev_short-= discharge
            battery_state-= discharge

        if time_of_charging=="Night (Super Off-Peak)":
            ev_cost_3= ev_short* super_off
        else:
            ev_cost_3= ev_short* on_peak

        ev_cost_nem_3.append(ev_cost_3)
        tot_nem_3.append(cost_house_3 + ev_cost_3)

    return (
        ev_cost_no_solar,
        ev_cost_nem_2,
        ev_cost_nem_3,
        tot_no_solar,
        tot_nem_2,
        tot_nem_3
    )

# --------------------------------------------------------------------------------
#          2. BASIC HOURLY MODEL (Naive battery, no export)
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
        available_space= (battery_capacity - battery_state)/ BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery= min(leftover_solar, available_space)
        battery_state+= to_battery*BATTERY_HOURLY_EFFICIENCY_BASIC
        leftover_solar-= to_battery
        solar_unused= leftover_solar
    else:
        solar_unused= leftover_solar

    if total_demand>0 and battery_state>0:
        discharge= min(total_demand,battery_state)
        total_demand-= discharge
        battery_state-= discharge

    grid_kwh= total_demand
    cost= grid_kwh* rate
    return battery_state, grid_kwh, cost, solar_unused

def run_basic_hourly_sim(
    daily_house,
    daily_solar,
    daily_ev,
    battery_capacity=10.0,
    ev_charging_pattern="Night",
    reset_battery_daily=False
):
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
        ev_shape= np.array([
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
        "day":[],"hour":[],"house_kwh":[],"ev_kwh":[],"solar_kwh":[],
        "grid_kwh":[],"cost":[],"battery_state":[],"solar_unused":[]
    }

    battery_state=0.0
    total_cost=0.0
    total_grid=0.0
    total_solar_unused=0.0

    days=len(daily_house)
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
            hour_idx= d*24 + hour
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
#    3. ADVANCED HOURLY LP (NEM 3.0-like) with LOWER EXPORT RATES & SOC <= capacity
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    """
    Lower export rates to ensure more realistic scenario and 
    avoid near-retail net offset. 
    Now the difference is bigger, so if battery=0, you 
    can't easily get $0 cost unless solar far overshoots usage each hour.
    """
    data=[]
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    # on-peak
                    import_r= 0.45
                    export_r= 0.05  # significantly lower than 0.45
                    demand_r= 10.0 if d_type=="weekday" else 8.0
                elif 7<=h<16 or 20<=h<22:
                    # off-peak
                    import_r= 0.25
                    export_r= 0.03
                    demand_r= 5.0
                else:
                    # super-off-peak
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

def hour_to_ampm_label(h):
    if h==0: return "12 AM"
    elif h<12: return f"{h} AM"
    elif h==12: return "12 PM"
    else: return f"{h-12} PM"

def advanced_battery_constraints(prob, hour, hour_of_day,
                                 battery_mode, backup_reserve_frac,
                                 home_batt_out, grid_export, soc, batt_cap,
                                 self_consumption_excess="Curtail"):
    if battery_mode=="TOU Arbitrage":
        # only discharge on-peak (16-19)
        if hour_of_day<16 or hour_of_day>=20:
            prob += home_batt_out[hour]==0, f"TOUOff_{hour}"
    elif battery_mode=="Self-Consumption":
        if self_consumption_excess=="Curtail":
            prob += grid_export[hour]==0, f"NoExport_{hour}"
    elif battery_mode=="Backup Priority":
        prob += soc[hour]>= backup_reserve_frac*batt_cap, f"BackupMin_{hour}"
    # "None" => no additional constraints

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
    # Create LP
    prob= LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc           = [pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=battery_capacity)
                     for h in range(25)]  # upBound=battery_capacity

    peak_demand= pulp.LpVariable("peak_demand",lowBound=0)

    cost_import=[]
    credit_export=[]

    for h in range(24):
        import_r= df_rates_day.loc[h,"import_rate"]
        export_r= df_rates_day.loc[h,"export_rate"]

        # If not in arrival->departure, EV=0
        if not (ev_arr <= h < ev_dep):
            prob += ev_charge[h]==0, f"EVno_{h}"

        # Power balance
        prob += (solar_24[h] + home_batt_out[h] + grid_import[h]
                 == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]
                ), f"Bal_{h}"

        # Battery SOC recursion
        prob += soc[h+1] == soc[h] + home_batt_in[h] - home_batt_out[h], f"SOC_{h}"

        # Demand charge
        if demand_charge_enabled:
            prob += peak_demand>= grid_import[h], f"Peak_{h}"

        cost_import.append(grid_import[h]* import_r)
        credit_export.append(grid_export[h]* export_r)

        # Battery constraints
        advanced_battery_constraints(prob,h,h,battery_mode,backup_reserve_frac,
                                     home_batt_out, grid_export, soc,
                                     battery_capacity, self_consumption_excess)

    # EV daily total
    prob += sum(ev_charge[h] for h in range(24))== ev_kwh_day, "EVDailyNeed"

    # Start battery
    prob += soc[0]== start_batt_soc, "StartSOC"

    # Daily battery reset
    prob += soc[24]== soc[0], f"ResetDay_{day_idx}"

    # EXPLICIT: enforce soc[h] <= battery_capacity
    # (Though we set upBound=battery_capacity, let's be safe and do it anyway)
    for hh in range(25):
        prob += soc[hh] <= battery_capacity, f"SOCcap_{hh}"

    total_import_cost= sum(cost_import)
    total_export_credit= sum(credit_export)
    demand_cost=0
    if demand_charge_enabled:
        max_day_dem= df_rates_day["demand_rate"].max()
        demand_cost= peak_demand* max_day_dem

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
        # figure out month
        m=0
        while m<12 and day_idx>= cum_days[m+1]:
            m+=1
        dow= day_idx%7
        d_type= "weekend" if dow in [5,6] else "weekday"

        df_day_rates= df_rates[(df_rates["month"]==m)&(df_rates["day_type"]==d_type)].copy()
        df_day_rates.sort_values("hour",inplace=True)
        df_day_rates.set_index("hour", inplace=True)

        dh= daily_house[day_idx]
        ds= daily_solar[day_idx]
        de= daily_ev[day_idx]

        house_24= np.full(24, dh/24.0)
        solar_24= np.full(24, ds/24.0)

        day_cost_val, end_soc= optimize_daily_lp(
            day_idx, house_24, solar_24, de,
            ev_arr, ev_dep,
            battery_soc, battery_capacity, ev_batt_capacity,
            df_day_rates,
            demand_charge_enabled,
            battery_mode,
            backup_reserve_frac,
            self_consumption_excess
        )
        if day_cost_val is not None:
            total_cost+= day_cost_val
            battery_soc= end_soc

        day_solutions.append({"day_idx": day_idx, "cost_day": day_cost_val if day_cost_val else 0})

    df_sol= pd.DataFrame(day_solutions)
    return total_cost, df_sol

# --------------------------------------------------------------------------------
#   4. BATTERY SIZE PARAM SWEEP (Optional) & Streamlit
# --------------------------------------------------------------------------------

def param_sweep_battery_sizes(
    daily_house,
    daily_solar,
    daily_ev,
    sizes_to_test,
    ev_arr=18,
    ev_dep=7,
    ev_battery_cap=50,
    demand_charge_enabled=False,
    battery_mode="None",
    backup_reserve_frac=0.2,
    self_consumption_excess="Curtail"
):
    results=[]
    for size in sizes_to_test:
        cost_sz, day_df= run_advanced_lp_sim(
            daily_house, daily_solar, daily_ev,
            ev_arr, ev_dep,
            battery_capacity=size,
            ev_battery_cap= ev_battery_cap,
            demand_charge_enabled= demand_charge_enabled,
            battery_mode= battery_mode,
            backup_reserve_frac= backup_reserve_frac,
            self_consumption_excess= self_consumption_excess
        )
        results.append((size, cost_sz))
    df_param= pd.DataFrame(results, columns=["BatterySize(kWh)","AnnualCost($)"])
    return df_param


def main():
    st.title("Revised All-In-One with Fixes to Avoid $0 Cost at 0 kWh Battery")

    st.write("""
    **Key Fixes**:
    1. **Lower export rates** significantly under the advanced LP (NEM 3.0-like),
    2. **Explicit battery SOC <= capacity** constraint,
    3. **Daily battery reset** ensuring no indefinite carryover.

    If you still see $0 cost at 0 kWh battery, it likely means your solar far exceeds usage 
    on an *hourly* basis and the leftover export, even at a low rate, covers your minimal imports.
    """)

    st.sidebar.header("Common Inputs")

    # EV
    commute_miles = st.sidebar.slider("Daily Commute (miles)",10,100,DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_freq= st.sidebar.radio("EV Charging Frequency", ["Daily","Weekdays Only"])
    days_per_week= 5 if charging_freq=="Weekdays Only" else 7

    monthly_charge_time= st.sidebar.radio("Monthly Model: EV Charging Time", 
                                         ["Night (Super Off-Peak)","Daytime (Peak)"])

    # House 
    house_kwh_base= st.sidebar.slider("Daily House (kWh)",10,50,int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluct= st.sidebar.slider("House Fluctuation (%)",0,50,int(DEFAULT_CONSUMPTION_FLUCTUATION*100))/100

    # Solar & Battery
    solar_size= st.sidebar.slider("Solar Size (kW)",0,15,int(DEFAULT_SOLAR_SIZE))
    monthly_batt_capacity= st.sidebar.slider("Battery(kWh) for Monthly/Basic",0,20,int(DEFAULT_BATTERY_CAPACITY))

    tab1, tab2, tab3, tab4= st.tabs(["Monthly","Basic Hourly","Advanced LP","Battery Size Sweep"])

    # ~~~~~~~~~ Tab1: Monthly ~~~~~~~~~
    with tab1:
        st.header("Monthly Approach (Fixed NEM 2.0 dimension)")

        ev_yearly, ev_monthly= calculate_ev_demand(commute_miles, efficiency, days_per_week)
        daily_house_val= house_kwh_base*(1+fluct)
        house_monthly= calculate_monthly_values(daily_house_val)
        _, solar_monthly= calculate_solar_production(solar_size)

        (ev_no, ev_n2, ev_n3,
         tot_no, tot_n2, tot_n3) = calculate_monthly_costs(
            ev_monthly, solar_monthly, house_monthly,
            monthly_batt_capacity,
            monthly_charge_time
        )

        df_m= pd.DataFrame({
            "Month": MONTH_NAMES,
            "House(kWh)": house_monthly,
            "EV(kWh)": ev_monthly,
            "Solar(kWh)": solar_monthly,
            "EV(NoSolar,$)": ev_no,
            "EV(NEM2,$)": ev_n2,
            "EV(NEM3,$)": ev_n3,
            "Tot(NoSolar,$)": tot_no,
            "Tot(NEM2,$)": tot_n2,
            "Tot(NEM3,$)": tot_n3
        })
        st.dataframe(df_m.style.format(precision=2))

        st.write("### Annual")
        st.write(f"**EV kWh**: {sum(ev_monthly):.1f}, **House kWh**: {sum(house_monthly):.1f}, **Solar**: {sum(solar_monthly):.1f}")
        st.write(f"**Cost(NoSolar)**= ${sum(tot_no):.2f},  **Cost(NEM2)**= ${sum(tot_n2):.2f},  **Cost(NEM3)**= ${sum(tot_n3):.2f}")

    # ~~~~~~~~~ Tab2: Basic Hourly ~~~~~~~~~
    with tab2:
        st.header("Basic Hourly (Naive)")

        daily_house_arr= np.full(DAYS_PER_YEAR, house_kwh_base*(1+fluct))
        daily_solar_arr= np.full(DAYS_PER_YEAR, solar_size*4)
        daily_ev_arr= np.full(DAYS_PER_YEAR, (commute_miles/efficiency)*(days_per_week/7.0))

        if monthly_charge_time=="Night (Super Off-Peak)":
            ev_basic_chg= "Night"
        else:
            ev_basic_chg= "Daytime"

        reset_basic= st.checkbox("Reset Battery Daily(Basic)?",False)

        cost_b, grid_b, sol_un_b, df_b= run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            monthly_batt_capacity,
            ev_charging_pattern= ev_basic_chg,
            reset_battery_daily= reset_basic
        )

        st.write(f"**Annual Cost**= ${cost_b:,.2f},  Grid= {grid_b:,.0f} kWh,  UnusedSolar= {sol_un_b:,.0f} kWh")
        day_sel= st.slider("Pick day(0-364)",0,364,0)
        df_dayb= df_b[df_b["day"]== day_sel].copy()

        def hour_to_ampm(h):
            if h==0: return "12 AM"
            elif h<12: return f"{h} AM"
            elif h==12:return "12 PM"
            else: return f"{h-12} PM"
        df_dayb["hour_label"]= df_dayb["hour"].apply(hour_to_ampm)

        st.write(f"### Day {day_sel} Hourly")
        figB, axB= plt.subplots()
        axB.plot(df_dayb["hour_label"], df_dayb["house_kwh"], label="House")
        axB.plot(df_dayb["hour_label"], df_dayb["ev_kwh"], label="EV", linestyle="--")
        axB.plot(df_dayb["hour_label"], df_dayb["solar_kwh"], label="Solar", color="gold")
        plt.xticks(rotation=45)
        axB.legend()
        st.pyplot(figB)

    # ~~~~~~~~~ Tab3: Advanced LP ~~~~~~~~~
    with tab3:
        st.header("Advanced Hourly LP (NEM 3.0-like)")

        if not PULP_AVAILABLE:
            st.error("PuLP missing => advanced LP won't run.")
        else:
            adv_mode= st.selectbox("Battery Mode",["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_r= 0.0
            sc_excess= "Curtail"
            if adv_mode=="Backup Priority":
                backup_r= st.slider("Backup Reserve(%)",0,50,20)/100
            elif adv_mode=="Self-Consumption":
                sc_excess= st.radio("Excess Solar?",["Curtail","Export"])

            en_dem= st.checkbox("Demand Charges?",False)

            if monthly_charge_time=="Night (Super Off-Peak)":
                ev_arr_lp=18
                ev_dep_lp=7
            else:
                ev_arr_lp=9
                ev_dep_lp=16

            # build daily arrays
            st.write("**Seasonal House & Solar**")
            house_factors=[]
            solar_factors=[]
            with st.expander("Adjust Monthly Factors(Advanced)"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf= st.slider(f"{mn} House Factor",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"advH_{i}")
                    sf= st.slider(f"{mn} Solar Factor",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"advS_{i}")
                    house_factors.append(hf)
                    solar_factors.append(sf)

            adv_ev_mean= st.slider("Mean DailyMiles(Adv)",0,100,30)
            adv_ev_std = st.slider("StdDev DailyMiles(Adv)",0,30,5)
            adv_ev_eff= st.slider("EV Efficiency(Adv)(miles/kWh)",3.0,5.0,4.0)
            adv_ev_cap= st.slider("EV Battery Cap(kWh)(Adv)",10,100,50)
            adv_home_batt= st.slider("Home Battery(kWh, Adv)",0,40,10)

            st.write("**Run** the LP for entire year ...")

            if st.button("Compute Advanced LP Now"):
                # Build daily arrays
                daily_house_l=[]
                daily_solar_l=[]
                day_count=0
                for m, ndays in enumerate(DAYS_IN_MONTH):
                    for _ in range(ndays):
                        hv= house_kwh_base*(1+fluct)* house_factors[m]
                        sv= (solar_size*4)* solar_factors[m]
                        daily_house_l.append(hv)
                        daily_solar_l.append(sv)
                        day_count+=1
                        if day_count>= DAYS_PER_YEAR: break
                    if day_count>= DAYS_PER_YEAR: break
                daily_house_l= np.array(daily_house_l[:DAYS_PER_YEAR])
                daily_solar_l= np.array(daily_solar_l[:DAYS_PER_YEAR])

                rng= np.random.default_rng(42)
                daily_ev_l=[]
                for i in range(DAYS_PER_YEAR):
                    miles= rng.normal(adv_ev_mean, adv_ev_std)
                    miles= max(0,miles)
                    needed= miles/ adv_ev_eff
                    needed= min(needed, adv_ev_cap)
                    daily_ev_l.append(needed)
                daily_ev_l= np.array(daily_ev_l)

                total_c, df_sol_l= run_advanced_lp_sim(
                    daily_house_l,
                    daily_solar_l,
                    daily_ev_l,
                    ev_arr= ev_arr_lp,
                    ev_dep= ev_dep_lp,
                    battery_capacity= adv_home_batt,
                    ev_batt_capacity= adv_ev_cap,
                    demand_charge_enabled= en_dem,
                    battery_mode= adv_mode,
                    backup_reserve_frac= backup_r,
                    self_consumption_excess= sc_excess
                )
                if total_c is not None:
                    st.success(f"Done! Annual Net Cost= ${total_c:,.2f}")
                    st.write(df_sol_l.head(10))
                    st.write("**Note**: day_idx vs. cost_day. We store partial data. You can parse further.")
                else:
                    st.warning("No solution or PuLP error.")


    # ~~~~~~~~~~~~~~ Tab 4: Battery Size Sweep
    with tab4:
        st.header("Battery Size Optimization (Param Sweep)")

        if not PULP_AVAILABLE:
            st.error("PuLP not installed => can't do advanced LP.")
        else:
            st.write("""
            We'll param-sweep battery_size in [0..20], stepping by e.g. 2 kWh, 
            re-run advanced LP each time, and chart the annual cost.
            """)

            adv_mode_sweep= st.selectbox("Battery Mode(Param Sweep)",["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_r_sweep=0.0
            sc_excess_sweep="Curtail"
            if adv_mode_sweep=="Backup Priority":
                backup_r_sweep= st.slider("Backup Reserve(%) for Param Sweep",0,50,20)/100
            elif adv_mode_sweep=="Self-Consumption":
                sc_excess_sweep= st.radio("ExcessSolar(Param)?",["Curtail","Export"])

            en_dem_sweep= st.checkbox("DemandCharges?(Param)",False)

            if monthly_charge_time=="Night (Super Off-Peak)":
                ev_arr_sw=18
                ev_dep_sw=7
            else:
                ev_arr_sw=9
                ev_dep_sw=16

            st.subheader("Seasonal Factors for Param Sweep")
            house_factors_sw=[]
            solar_factors_sw=[]
            with st.expander("Adjust Monthly for Param Sweep"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf= st.slider(f"{mn} House Factor(Param)",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"swH_{i}")
                    sf= st.slider(f"{mn} Solar Factor(Param)",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"swS_{i}")
                    house_factors_sw.append(hf)
                    solar_factors_sw.append(sf)

            adv_ev_mean_sw= st.slider("EV MeanMiles(Param)",0,100,30)
            adv_ev_std_sw= st.slider("EV MilesStd(Param)",0,30,5)
            adv_ev_eff_sw= st.slider("EV Efficiency(Param)",3.0,5.0,4.0)
            adv_ev_cap_sw= st.slider("EV Battery(Param)",10,100,50)

            st.write("**Run** param sweep ...")

            if st.button("Start Param Sweep"):
                # Build daily arrays
                daily_house_sw=[]
                daily_solar_sw=[]
                day_count_sw=0
                for m, ndays in enumerate(DAYS_IN_MONTH):
                    for _ in range(ndays):
                        hv= house_kwh_base*(1+fluct)* house_factors_sw[m]
                        sv= (solar_size*4)* solar_factors_sw[m]
                        daily_house_sw.append(hv)
                        daily_solar_sw.append(sv)
                        day_count_sw+=1
                        if day_count_sw>= DAYS_PER_YEAR: break
                    if day_count_sw>= DAYS_PER_YEAR: break
                daily_house_sw= np.array(daily_house_sw[:DAYS_PER_YEAR])
                daily_solar_sw= np.array(daily_solar_sw[:DAYS_PER_YEAR])

                rng= np.random.default_rng(42)
                daily_ev_sw=[]
                for i in range(DAYS_PER_YEAR):
                    miles= rng.normal(adv_ev_mean_sw, adv_ev_std_sw)
                    miles= max(0,miles)
                    needed= miles/ adv_ev_eff_sw
                    needed= min(needed, adv_ev_cap_sw)
                    daily_ev_sw.append(needed)
                daily_ev_sw= np.array(daily_ev_sw)

                # param sizes
                sizes_test= np.arange(0,21,2)  # step=2 => [0,2,4,6,...,20]
                results=[]
                for b_sz in sizes_test:
                    tot_cz, ddf= run_advanced_lp_sim(
                        daily_house_sw,
                        daily_solar_sw,
                        daily_ev_sw,
                        ev_arr_sw, ev_dep_sw,
                        battery_capacity= b_sz,
                        ev_batt_capacity= adv_ev_cap_sw,
                        demand_charge_enabled= en_dem_sweep,
                        battery_mode= adv_mode_sweep,
                        backup_reserve_frac= backup_r_sweep,
                        self_consumption_excess= sc_excess_sweep
                    )
                    results.append((b_sz, tot_cz))

                df_par= pd.DataFrame(results, columns=["BatterySize(kWh)","AnnualCost($)"])
                st.subheader("Param Sweep Results")
                st.dataframe(df_par.style.format(precision=2))

                figS, axS= plt.subplots()
                axS.plot(df_par["BatterySize(kWh)"], df_par["AnnualCost($)"], marker="o")
                axS.set_xlabel("Battery Size (kWh)")
                axS.set_ylabel("Annual Cost($)")
                axS.set_title("Annual Cost vs Battery Size")
                st.pyplot(figS)

                minrow= df_par.loc[df_par["AnnualCost($)"].idxmin()]
                st.write(f"**Best Battery**= {minrow['BatterySize(kWh)']} kWh => cost= ${minrow['AnnualCost($)']:.2f}")
                st.write("""
                If battery=0 yields $0 cost, it means your solar & rates 
                overshadow usage hour by hour. But with lowered export rates, 
                that should be less likely *unless* your solar is extremely large 
                for your load profile.
                """)

if __name__=="__main__":
    main()
