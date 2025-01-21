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
#                          GLOBAL CONSTANTS
# --------------------------------------------------------------------------------

DEFAULT_COMMUTE_MILES = 30
DEFAULT_EFFICIENCY = {"Model Y": 3.5, "Model 3": 4.0}
DEFAULT_BATTERY_CAPACITY = 10  # kWh (for monthly/basic default)
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
    "off_peak_hours": list(range(7, 16)) + [21,22],
    "super_off_peak_hours": list(range(0, 7)) + [23],
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
    ev_cost_no_solar = []
    ev_cost_nem_2 = []
    ev_cost_nem_3 = []

    tot_no_solar = []
    tot_nem_2 = []
    tot_nem_3 = []

    battery_state= 0.0

    for m in range(12):
        if m in SUMMER_MONTHS:
            on_peak= 0.45
            off_peak= 0.25
            super_off= 0.12
        else:
            on_peak= 0.35
            off_peak= 0.20
            super_off= 0.10

        # (1) No Solar
        cost_house_ns = house_monthly[m]* off_peak
        cost_ev_ns    = ev_monthly[m]* super_off
        ev_cost_no_solar.append(cost_ev_ns)
        tot_no_solar.append(cost_house_ns + cost_ev_ns)

        # (2) NEM 2.0
        house_cost_n2 = house_monthly[m]* off_peak
        leftover_solar_kwh = max(0, solar_monthly[m]- house_monthly[m])
        ev_kwh= ev_monthly[m]

        if time_of_charging=="Night (Super Off-Peak)":
            ev_rate= super_off
        else:
            ev_rate= on_peak

        if leftover_solar_kwh>= ev_kwh:
            leftover_export_kwh= leftover_solar_kwh- ev_kwh
            ev_cost2_val= 0.0
            # leftover export credited at off_peak
            credit_n2_val= leftover_export_kwh* off_peak
        else:
            offset_kwh= leftover_solar_kwh
            leftover_export_kwh= 0.0
            ev_grid_kwh= ev_kwh- offset_kwh
            ev_cost2_val= ev_grid_kwh* ev_rate
            credit_n2_val= 0.0

        cost_n2 = house_cost_n2 + ev_cost2_val - credit_n2_val
        ev_cost_nem_2.append(ev_cost2_val)
        tot_nem_2.append(cost_n2)

        # (3) NEM 3.0 + Battery (naive monthly)
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
#           2. BASIC HOURLY MODEL (Naive Battery, No Export)
# --------------------------------------------------------------------------------

def classify_tou_basic(hour):
    if hour in HOUR_TOU_SCHEDULE_BASIC["on_peak_hours"]:
        return "on_peak"
    elif hour in HOUR_TOU_SCHEDULE_BASIC["off_peak_hours"]:
        return "off_peak"
    else:
        return "super_off_peak"

def simulate_hour_basic(hour_idx, solar_kwh, house_kwh, ev_kwh, battery_state, battery_capacity):
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
        available_space= (battery_capacity- battery_state)/ BATTERY_HOURLY_EFFICIENCY_BASIC
        to_battery= min(leftover_solar, available_space)
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

    for d in range(len(daily_house)):
        if reset_battery_daily:
            battery_state=0.0

        dh= daily_house[d]
        ds= daily_solar[d]
        de= daily_ev[d]

        h_24= dh*house_shape
        s_24= ds*solar_shape
        e_24= de*ev_shape

        for hour in range(24):
            hour_idx= d*24+ hour
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
#        3. ADVANCED HOURLY LP (NEM 3.0-like)
# --------------------------------------------------------------------------------

def generate_utility_rate_schedule():
    """
    Lower export rates to ensure export < import in each time block.
    """
    data=[]
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    import_r=0.45
                    export_r=0.07
                    demand_r=10.0 if d_type=="weekday" else 8.0
                elif 7<=h<16 or 20<=h<22:
                    import_r=0.25
                    export_r=0.05
                    demand_r=5.0
                else:
                    import_r=0.12
                    export_r=0.03
                    demand_r=2.0
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
    elif h==12:return "12 PM"
    else: return f"{h-12} PM"

def advanced_battery_constraints(prob, hour, hour_of_day,
                                 battery_mode, backup_reserve_frac,
                                 home_batt_out, grid_export, soc, batt_cap,
                                 self_consumption_excess="Curtail"):
    if battery_mode=="TOU Arbitrage":
        if hour_of_day<16 or hour_of_day>=20:
            prob += home_batt_out[hour]==0, f"TOUOff_{hour}"
    elif battery_mode=="Self-Consumption":
        if self_consumption_excess=="Curtail":
            prob += grid_export[hour]==0, f"SCNoExport_{hour}"
    elif battery_mode=="Backup Priority":
        prob += soc[hour]>= backup_reserve_frac*batt_cap, f"BackupMin_{hour}"
    # "None" => no extra constraints

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
    prob= LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc           = [pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=battery_capacity) for h in range(25)]

    peak_demand= pulp.LpVariable("peak_demand",lowBound=0)
    cost_import=[]
    credit_export=[]

    for h in range(24):
        import_r= df_rates_day.loc[h,"import_rate"]
        export_r= df_rates_day.loc[h,"export_rate"]

        if not (ev_arr <= h < ev_dep):
            prob += ev_charge[h]==0, f"EVNo_{h}"

        prob += (solar_24[h] + home_batt_out[h] + grid_import[h]
                 == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h]), f"Bal_{h}"

        prob += soc[h+1] == soc[h] + home_batt_in[h] - home_batt_out[h], f"SOC_{h}"

        if demand_charge_enabled:
            prob += peak_demand>= grid_import[h], f"Peak_{h}"

        cost_import.append(grid_import[h]* import_r)
        credit_export.append(grid_export[h]* export_r)

        advanced_battery_constraints(prob,h,h,
                                     battery_mode, backup_reserve_frac,
                                     home_batt_out, grid_export, soc,
                                     battery_capacity,
                                     self_consumption_excess)

    # EV daily requirement
    prob += sum(ev_charge[h] for h in range(24))== ev_kwh_day, "EVDayReq"

    # Start battery
    prob += soc[0]== start_batt_soc, "StartSOC"

    # Daily reset
    prob += soc[24] == soc[0], f"ResetDay_{day_idx}"

    total_import_cost= sum(cost_import)
    total_export_credit= sum(credit_export)
    demand_cost=0
    if demand_charge_enabled:
        max_day_demand_rate= df_rates_day["demand_rate"].max()
        demand_cost= peak_demand* max_day_demand_rate

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
        df_day_rates.sort_values("hour", inplace=True)
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

        # store partial data
        day_solutions.append({"day_idx": day_idx, "cost_day": day_cost_val if day_cost_val else 0})

    df_sol= pd.DataFrame(day_solutions)
    return total_cost, df_sol


# --------------------------------------------------------------------------------
# 4. Battery Size Optimization Tab (param sweep)
# --------------------------------------------------------------------------------

def param_sweep_battery_sizes(
    daily_house,
    daily_solar,
    daily_ev,
    sizes_to_test,
    ev_arr=18,
    ev_dep=7,
    ev_batt_capacity=50,
    demand_charge_enabled=False,
    battery_mode="None",
    backup_reserve_frac=0.2,
    self_consumption_excess="Curtail"
):
    """
    For each battery size in sizes_to_test, run the advanced LP 
    and store the total cost. Return a DataFrame.
    """
    results=[]
    for size in sizes_to_test:
        total_c, _ = run_advanced_lp_sim(
            daily_house,
            daily_solar,
            daily_ev,
            ev_arr, ev_dep,
            battery_capacity=size,
            ev_batt_capacity=ev_batt_capacity,
            demand_charge_enabled=demand_charge_enabled,
            battery_mode=battery_mode,
            backup_reserve_frac=backup_reserve_frac,
            self_consumption_excess=self_consumption_excess
        )
        results.append((size, total_c))

    df_param = pd.DataFrame(results, columns=["BatterySize(kWh)", "AnnualCost($)"])
    return df_param


# --------------------------------------------------------------------------------
# STREAMLIT MAIN
# --------------------------------------------------------------------------------

def main():
    st.title("All-In-One App + Battery Size Optimization Param Sweep")

    st.write("""
    This single app has:
    1. **Monthly Approach** (NEM 2.0 vs. naive NEM 3.0),
    2. **Basic Hourly** (no export credit),
    3. **Advanced Hourly LP** (time-based import/export rates, daily battery reset),
    4. **Battery Size Optimization** (param sweep) for the advanced LP model.
    """)

    st.sidebar.header("Common Inputs")

    # EV
    commute_miles = st.sidebar.slider("Daily Commute (miles)",10,100,DEFAULT_COMMUTE_MILES)
    ev_model = st.sidebar.selectbox("EV Model", list(DEFAULT_EFFICIENCY.keys()))
    efficiency = DEFAULT_EFFICIENCY[ev_model]
    charging_freq= st.sidebar.radio("EV Charging Frequency", ["Daily","Weekdays Only"])
    days_per_week= 5 if charging_freq=="Weekdays Only" else 7

    monthly_charge_time= st.sidebar.radio("Monthly Model: EV Charging Time", 
                                         ["Night (Super Off-Peak)","Daytime (Peak)"],
                                         index=0)

    # House 
    house_kwh_base= st.sidebar.slider("Daily House (kWh)",10,50,int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluct= st.sidebar.slider("House Fluctuation (%)",0,50,int(DEFAULT_CONSUMPTION_FLUCTUATION*100))/100

    # Solar + Battery
    solar_size= st.sidebar.slider("Solar Size (kW)",0,15,int(DEFAULT_SOLAR_SIZE))
    monthly_batt_capacity= st.sidebar.slider("Battery (kWh for Monthly/Basic)",0,20,int(DEFAULT_BATTERY_CAPACITY))

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Approach","Basic Hourly","Advanced Hourly LP","Battery Size Optimization"])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TAB 1: MONTHLY
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab1:
        st.header("Monthly Approach (NEM 2.0 vs NEM 3.0)")

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
            "EV NoSolar($)": ev_no,
            "EV NEM2($)": ev_n2,
            "EV NEM3($)": ev_n3,
            "Tot NoSolar($)": tot_no,
            "Tot NEM2($)": tot_n2,
            "Tot NEM3($)": tot_n3
        })
        st.dataframe(df_m.style.format(precision=2))

        st.write("### Annual Summaries")
        st.write(f"**Annual EV kWh**: {sum(ev_monthly):.1f}")
        st.write(f"**Annual House kWh**: {sum(house_monthly):.1f}")
        st.write(f"**Annual Solar**: {sum(solar_monthly):.1f} kWh")
        st.write(f"**Total Cost (No Solar)**: ${sum(tot_no):.2f}")
        st.write(f"**Total Cost (NEM 2.0)**: ${sum(tot_n2):.2f}")
        st.write(f"**Total Cost (NEM 3.0 + Battery)**: ${sum(tot_n3):.2f}")

        st.write("""
        **Note**: NEM 2.0 net offsets the entire month, possibly offsetting nighttime EV usage 
        with daytime solar. NEM 3.0 + Battery in naive monthly form might not offset night usage 
        if you choose "Night" charging.
        """)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TAB 2: BASIC HOURLY
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab2:
        st.header("Basic Hourly (No Export)")

        daily_house_arr= np.full(DAYS_PER_YEAR, house_kwh_base*(1+fluct))
        daily_solar_arr= np.full(DAYS_PER_YEAR, solar_size*4)
        daily_ev_arr= np.full(DAYS_PER_YEAR, (commute_miles/efficiency)*(days_per_week/7.0))

        # Convert monthly_charge_time => "Night"/"Daytime" for basic
        if monthly_charge_time=="Night (Super Off-Peak)":
            ev_pattern= "Night"
        else:
            ev_pattern= "Daytime"

        reset_basic= st.checkbox("Reset Battery Daily (Basic)?",False)

        cost_b, grid_b, sol_un_b, df_b= run_basic_hourly_sim(
            daily_house_arr,
            daily_solar_arr,
            daily_ev_arr,
            battery_capacity=monthly_batt_capacity,
            ev_charging_pattern= ev_pattern,
            reset_battery_daily= reset_basic
        )

        st.write(f"**Total Annual Cost**: ${cost_b:,.2f}")
        st.write(f"**Grid Usage**: {grid_b:,.0f} kWh")
        st.write(f"**Unused Solar**: {sol_un_b:,.0f} kWh")

        day_choice= st.slider("Pick a Day(0-364)",0,364,0)
        df_day_b= df_b[df_b["day"]== day_choice].copy()

        def hour_to_ampm(h):
            if h==0: return "12 AM"
            elif h<12: return f"{h} AM"
            elif h==12:return "12 PM"
            else: return f"{h-12} PM"
        df_day_b["hour_label"]= df_day_b["hour"].apply(hour_to_ampm)

        st.subheader(f"Hourly Breakdown - Day {day_choice}")
        figB, axB= plt.subplots()
        axB.plot(df_day_b["hour_label"], df_day_b["house_kwh"], label="House")
        axB.plot(df_day_b["hour_label"], df_day_b["ev_kwh"], label="EV", linestyle="--")
        axB.plot(df_day_b["hour_label"], df_day_b["solar_kwh"], label="Solar", color="gold")
        plt.xticks(rotation=45)
        axB.legend()
        st.pyplot(figB)

        st.write("Battery & Grid")
        figB2, axB2= plt.subplots()
        axB2.plot(df_day_b["hour_label"], df_day_b["battery_state"], color="green", label="Battery(kWh)")
        axB2.set_ylabel("Battery(kWh)")
        plt.xticks(rotation=45)
        axB2.legend(loc="upper left")

        axB3= axB2.twinx()
        axB3.plot(df_day_b["hour_label"], df_day_b["grid_kwh"], color="red", label="Grid(kWh)")
        axB3.set_ylabel("Grid(kWh)")
        axB3.legend(loc="upper right")
        st.pyplot(figB2)

        st.write("""
        **Note**: There's no net export credit. Any leftover solar in an hour is just "unused" 
        unless it goes into the battery.
        """)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TAB 3: ADVANCED HOURLY LP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab3:
        st.header("Advanced Hourly LP (NEM 3.0-like, daily battery reset)")

        if not PULP_AVAILABLE:
            st.error("PuLP not installed => advanced LP won't run.")
        else:
            st.write("""
            **Modes**: 
            - None, 
            - TOU Arbitrage (only discharge on-peak), 
            - Self-Consumption (curtail or export leftover), 
            - Backup Priority (keep min SOC).
            
            We reset the battery daily (soc[24]= soc[0]) to avoid indefinite carryover.
            """)

            adv_mode= st.selectbox("Battery Mode",["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_res=0.0
            sc_excess="Curtail"
            if adv_mode=="Backup Priority":
                backup_res= st.slider("Backup Reserve (%)",0,50,20)/100
            elif adv_mode=="Self-Consumption":
                sc_excess= st.radio("Excess Solar in Self-Cons?",["Curtail","Export"])

            enable_demand= st.checkbox("Enable Demand Charges?",False)

            # unify "Night" vs "Daytime" from monthly side
            if monthly_charge_time=="Night (Super Off-Peak)":
                ev_arr=18
                ev_dep=7
            else:
                ev_arr=9
                ev_dep=16

            st.subheader("Seasonal House & Solar Factors")
            house_factors=[]
            solar_factors=[]
            with st.expander("Adjust Monthly Factors"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf= st.slider(f"{mn} House Factor",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"adv_housef_{i}")
                    sf= st.slider(f"{mn} Solar Factor",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"adv_solarf_{i}")
                    house_factors.append(hf)
                    solar_factors.append(sf)

            st.subheader("Advanced EV & Battery")
            adv_ev_mean= st.slider("Mean Daily Miles(Adv EV)",0,100,30)
            adv_ev_std= st.slider("StdDev Daily Miles",0,30,5)
            adv_ev_eff= st.slider("EV Efficiency(miles/kWh)",3.0,5.0,4.0)
            adv_ev_cap= st.slider("EV Battery(kWh)",10,100,50)
            adv_home_batt= st.slider("Home Battery(kWh, Advanced LP)",0,40,10)

            st.write("**Running the LP** ...")

            # build daily arrays
            daily_house_2=[]
            daily_solar_2=[]
            day_count=0
            for m, ndays in enumerate(DAYS_IN_MONTH):
                for _ in range(ndays):
                    hv= house_kwh_base*(1+fluct)* house_factors[m]
                    sv= (solar_size*4)* solar_factors[m]
                    daily_house_2.append(hv)
                    daily_solar_2.append(sv)
                    day_count+=1
                    if day_count>= DAYS_PER_YEAR: break
                if day_count>= DAYS_PER_YEAR: break
            daily_house_2= np.array(daily_house_2[:DAYS_PER_YEAR])
            daily_solar_2= np.array(daily_solar_2[:DAYS_PER_YEAR])

            rng= np.random.default_rng(42)
            daily_ev_2=[]
            for i in range(DAYS_PER_YEAR):
                miles= rng.normal(adv_ev_mean, adv_ev_std)
                miles= max(0,miles)
                needed= miles/ adv_ev_eff
                needed= min(needed, adv_ev_cap)
                daily_ev_2.append(needed)
            daily_ev_2= np.array(daily_ev_2)

            adv_cost, df_adv_sol= run_advanced_lp_sim(
                daily_house_2,
                daily_solar_2,
                daily_ev_2,
                ev_arr, ev_dep,
                battery_capacity= adv_home_batt,
                ev_batt_capacity= adv_ev_cap,
                demand_charge_enabled= enable_demand,
                battery_mode= adv_mode,
                backup_reserve_frac= backup_res,
                self_consumption_excess= sc_excess
            )
            if adv_cost is not None:
                st.success(f"Done! Total Annual Net Cost = ${adv_cost:,.2f}")
                # df_adv_sol => day-based
                day_costs_lp= df_adv_sol
                st.write("**(We've only stored day-level cost)**")
                # The structure might differ in your code
                tot_import= None
                tot_export= None
                st.write("Param or partial data not fully stored. See param approach below.")
            else:
                st.warning("No solution or PuLP missing.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TAB 4: Battery Size Optimization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with tab4:
        st.header("Battery Size Optimization (Param Sweep)")

        if not PULP_AVAILABLE:
            st.error("PuLP not installed => can't do advanced LP.")
        else:
            st.write("""
            We run the advanced LP for multiple **battery sizes**, 
            then chart the total annual cost. 
            The user can see which battery size yields the lowest cost 
            (not factoring in capital cost).
            """)

            st.subheader("Seasonal House & Solar Factors (same logic)")
            house_factors_opt=[]
            solar_factors_opt=[]
            with st.expander("Adjust Monthly Factors for param sweep"):
                for i,mn in enumerate(MONTH_NAMES):
                    hf= st.slider(f"{mn} House Factor (Opt)",0.5,1.5,DEFAULT_LOAD_FACTORS[i],0.05, key=f"opt_housef_{i}")
                    sf= st.slider(f"{mn} Solar Factor (Opt)",0.5,1.5,DEFAULT_SOLAR_FACTORS[i],0.05, key=f"opt_solarf_{i}")
                    house_factors_opt.append(hf)
                    solar_factors_opt.append(sf)

            # unify "Night" vs "Daytime" from monthly side for EV arrival/depart
            if monthly_charge_time=="Night (Super Off-Peak)":
                ev_arr_opt= 18
                ev_dep_opt= 7
            else:
                ev_arr_opt= 9
                ev_dep_opt= 16

            adv_mode_opt= st.selectbox("Battery Mode (for param sweep)",
                                      ["None","TOU Arbitrage","Self-Consumption","Backup Priority"],
                                      index=0)
            backup_res_opt=0.0
            sc_excess_opt= "Curtail"
            if adv_mode_opt=="Backup Priority":
                backup_res_opt= st.slider("Backup Reserve(%) for Param?",0,50,20)/100
            elif adv_mode_opt=="Self-Consumption":
                sc_excess_opt= st.radio("Excess Solar in Self-Cons Param?",["Curtail","Export"])

            enable_demand_opt= st.checkbox("Enable Demand Charges? (Param)")

            st.subheader("Advanced EV for param sweep")
            adv_ev_mean_opt= st.slider("Mean Daily Miles(Param EV)",0,100,30)
            adv_ev_std_opt=  st.slider("StdDev Daily Miles(Param EV)",0,30,5)
            adv_ev_eff_opt=  st.slider("EV Efficiency(Param)(miles/kWh)",3.0,5.0,4.0)
            adv_ev_cap_opt=  st.slider("EV Battery(Param)(kWh)",10,100,50)

            # build daily arrays
            st.write("**Press 'Run Param Sweep' to test multiple battery sizes**")
            if st.button("Run Param Sweep"):
                # create daily arrays
                daily_house_opt=[]
                daily_solar_opt=[]
                day_count_opt= 0
                for m, ndays in enumerate(DAYS_IN_MONTH):
                    for _ in range(ndays):
                        hv= house_kwh_base*(1+fluct)* house_factors_opt[m]
                        sv= (solar_size*4)* solar_factors_opt[m]
                        daily_house_opt.append(hv)
                        daily_solar_opt.append(sv)
                        day_count_opt+=1
                        if day_count_opt>= DAYS_PER_YEAR: break
                    if day_count_opt>= DAYS_PER_YEAR: break

                daily_house_opt= np.array(daily_house_opt[:DAYS_PER_YEAR])
                daily_solar_opt= np.array(daily_solar_opt[:DAYS_PER_YEAR])

                rng= np.random.default_rng(42)
                daily_ev_opt=[]
                for i in range(DAYS_PER_YEAR):
                    miles= rng.normal(adv_ev_mean_opt, adv_ev_std_opt)
                    miles= max(0,miles)
                    needed= miles/ adv_ev_eff_opt
                    needed= min(needed, adv_ev_cap_opt)
                    daily_ev_opt.append(needed)
                daily_ev_opt= np.array(daily_ev_opt)

                # Battery sizes to test
                sizes_to_test= np.arange(0,21,2)  # 0,2,4,6,8,10,12,14,16,18,20
                df_param= []
                # We'll do a function param_sweep_battery_sizes
                # But let's just inline it here for clarity.
                results_sweep=[]
                from math import isinf

                # We'll re-use run_advanced_lp_sim
                for sz in sizes_to_test:
                    cost_sz, day_df = run_advanced_lp_sim(
                        daily_house_opt,
                        daily_solar_opt,
                        daily_ev_opt,
                        ev_arr_opt, ev_dep_opt,
                        battery_capacity= sz,
                        ev_batt_capacity= adv_ev_cap_opt,
                        demand_charge_enabled= enable_demand_opt,
                        battery_mode= adv_mode_opt,
                        backup_reserve_frac= backup_res_opt,
                        self_consumption_excess= sc_excess_opt
                    )
                    results_sweep.append((sz, cost_sz))

                df_param= pd.DataFrame(results_sweep, columns=["BatterySize(kWh)","AnnualCost($)"])
                st.subheader("Param Sweep Results")
                st.dataframe(df_param.style.format(precision=2))

                # Plot
                figP, axP= plt.subplots()
                axP.plot(df_param["BatterySize(kWh)"], df_param["AnnualCost($)"], marker="o")
                axP.set_xlabel("Battery Size (kWh)")
                axP.set_ylabel("Annual Cost ($)")
                axP.set_title("Annual Cost vs Battery Size (Param Sweep)")
                st.pyplot(figP)

                # Find min cost
                best_row= df_param.loc[df_param["AnnualCost($)"].idxmin()]
                st.write(f"**Best Battery Size**: {best_row['BatterySize(kWh)']} kWh => Cost= ${best_row['AnnualCost($)']:.2f}")
                st.write("""
                **Interpretation**: 
                - This approach doesn't factor in the capital cost of the battery. 
                - If you want ROI-based optimum, you'd add a cost for each kWh of battery 
                  and pick the size that yields the best net savings.
                """)

if __name__=="__main__":
    main()
