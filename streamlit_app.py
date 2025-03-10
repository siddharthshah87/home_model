import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Import from our local modules
from models.monthly_model import (
    calculate_ev_demand, calculate_solar_production, calculate_monthly_values,
    calculate_monthly_costs
)
from models.basic_hourly_model import (
    run_basic_hourly_sim
)
from models.advanced_lp_model import (
    run_advanced_lp_sim
)
from utils.param_sweep import (
    param_sweep_battery_sizes
)
from services.urdb_api import (
    fetch_urdb_plans_for_state, naive_parse_urdb_plan
)
from services.bill_parsing import (
    parse_utility_bill_pdf
)
from services.recommendation import (
    get_deepseek_recommendations
)

# Constants shared in app
DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DEFAULT_COMMUTE_MILES = 30
DEFAULT_SOLAR_SIZE = 7.5
DEFAULT_BATTERY_CAPACITY = 10
DEFAULT_HOUSEHOLD_CONSUMPTION = 17.8
DEFAULT_CONSUMPTION_FLUCTUATION = 0.2

def main():
    st.title("Energy App (Modular) + PDF Bill Parsing + Deepseek R1")

    # ~~~ 1) Bill Upload ~~~
    st.sidebar.header("Utility Bill (PDF) Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Utility Bill", type=["pdf"])
    user_bill_data = None

    if uploaded_file:
        # parse the PDF
        user_bill_data = parse_utility_bill_pdf(uploaded_file)
        if user_bill_data:
            st.sidebar.write("Parsed Bill Data:", user_bill_data)
        else:
            st.sidebar.warning("Could not parse or no relevant info found in PDF.")

    # ~~~ 2) URDB Plans (optional) ~~~
    with st.sidebar.expander("URDB Rate Plans"):
        states_list= ["AL","AR","AZ","CA","CO","CT","DE","FL","GA","HI",
                      "IA","ID","IL","IN","KS","KY","LA","MA","MD","ME",
                      "MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ",
                      "NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
                      "TN","TX","UT","VA","VT","WA","WI","WV","WY"]
        state_choice= st.selectbox("Pick State", states_list)
        if st.button("Fetch Plans"):
            plans= fetch_urdb_plans_for_state(state_choice)
            if len(plans)>0:
                st.session_state["urdb_plans"] = plans
                st.success(f"Found {len(plans)} plans for {state_choice}!")
            else:
                st.warning("No plans found or error.")

        chosen_plan=None
        if "urdb_plans" in st.session_state:
            plan_names= [p.get("name","(unnamed)") for p in st.session_state["urdb_plans"]]
            plan_idx= st.selectbox("Select URDB Plan", range(len(plan_names)),
                                   format_func=lambda i: plan_names[i][:60])
            if 0<= plan_idx< len(plan_names):
                chosen_plan= st.session_state["urdb_plans"][plan_idx]

        if chosen_plan:
            st.write("**Chosen Plan**:", chosen_plan.get("name","(No Name)"))
            st.write(chosen_plan)
            parsed_rate= naive_parse_urdb_plan(chosen_plan)
            st.session_state["urdb_rate_structure"] = parsed_rate
            st.success(f"Naive parse => {parsed_rate}")

    # ~~~ 3) Common Inputs ~~~
    st.sidebar.header("Common Inputs")
    commute_miles = st.sidebar.slider("Daily Commute(miles)", 10, 100, DEFAULT_COMMUTE_MILES)
    house_kwh_base= st.sidebar.slider("Daily House(kWh)", 10, 50, int(DEFAULT_HOUSEHOLD_CONSUMPTION))
    fluct= st.sidebar.slider("House Fluctuation(%)",0,50,int(DEFAULT_CONSUMPTION_FLUCTUATION*100))/100
    solar_size= st.sidebar.slider("Solar(kW)",0,15,int(DEFAULT_SOLAR_SIZE))

    charging_freq= st.sidebar.radio("EV Charging Frequency(Monthly/Basic)", ["Daily","Weekdays Only"])
    days_per_week= 5 if charging_freq=="Weekdays Only" else 7

    monthly_charge_time= st.sidebar.radio("Monthly Model: EV Charging Time",
                                         ["Night (Super Off-Peak)","Daytime (Peak)"])

    unified_batt_capacity= st.sidebar.slider("Battery(kWh, for all approaches, unless None)",0,20,int(DEFAULT_BATTERY_CAPACITY))

    # ~~~ 4) Tab Layout ~~~
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Monthly Approach","Basic Hourly","Advanced Hourly","Battery Size Sweep","Recommendations"
    ])

    with tab1:
        st.header("Monthly Approach")
        ev_yearly, ev_monthly = calculate_ev_demand(commute_miles, 4.0, days_per_week)  # or use the actual selected model
        daily_house_val = house_kwh_base*(1+fluct)
        house_monthly= calculate_monthly_values(daily_house_val)
        _, solar_monthly= calculate_solar_production(solar_size)

        ev_no, ev_n2, ev_n3, tot_no, tot_n2, tot_n3= calculate_monthly_costs(
            ev_monthly, solar_monthly, house_monthly,
            unified_batt_capacity,
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

        # Example Graph
        st.write("### Monthly EV Cost Comparison")
        fig1, ax1= plt.subplots()
        ax1.plot(df_m["Month"], df_m["EV(NoSolar,$)"], label="No Solar")
        ax1.plot(df_m["Month"], df_m["EV(NEM2,$)"], label="NEM2")
        ax1.plot(df_m["Month"], df_m["EV(NEM3,$)"], label="NEM3")
        ax1.legend()
        st.pyplot(fig1)

    with tab2:
        st.header("Basic Hourly")
        # replicate the basic approach
        days= DAYS_PER_YEAR
        daily_house_arr= np.full(days, house_kwh_base*(1+fluct))
        daily_solar_arr= np.full(days, solar_size*4)
        # EV usage
        daily_ev_arr= np.full(days, (commute_miles/4.0)*(days_per_week/7.0))  # or so

        reset_daily_batt= st.checkbox("Reset Battery Daily(Basic)?",False)
        if monthly_charge_time=="Night (Super Off-Peak)":
            ev_basic_pattern= "Night"
        else:
            ev_basic_pattern= "Daytime"

        cost_b, grid_b, sol_un_b, df_b= run_basic_hourly_sim(
            daily_house_arr, daily_solar_arr, daily_ev_arr,
            unified_batt_capacity,
            ev_basic_pattern,
            reset_battery_daily= reset_daily_batt
        )
        st.write(f"**Total Basic Hourly Cost**= ${cost_b:,.2f}")
        st.write(f"Grid= {grid_b:,.2f} kWh, UnusedSolar= {sol_un_b:,.2f} kWh")

        # let user pick day
        day_pick= st.slider("Pick Day(0..364)",0,364,0)
        df_day= df_b[df_b["day"]== day_pick].copy()
        st.write(df_day.head(20))
        # you can do the same plotting as in the original code

    with tab3:
        st.header("Advanced Hourly LP")

        if not PULP_AVAILABLE:
            st.error("PuLP not installed => can't do advanced LP.")
        else:
            adv_mode= st.selectbox("Battery Mode(Advanced)", ["None","TOU Arbitrage","Self-Consumption","Backup Priority"])
            backup_r=0.2
            en_demand= st.checkbox("Enable Demand Charges?",False)

            if st.button("Run Advanced LP"):
                # build daily arrays, run
                daily_house2= np.full(DAYS_PER_YEAR, house_kwh_base*(1+fluct))
                daily_solar2= np.full(DAYS_PER_YEAR, solar_size*4)
                daily_ev2= np.full(DAYS_PER_YEAR, (commute_miles/4.0)*(days_per_week/7.0))

                # use daytime or night to define ev_arr, ev_dep
                if monthly_charge_time=="Night (Super Off-Peak)":
                    ev_arr= 18; ev_dep=7
                else:
                    ev_arr= 9; ev_dep=16

                cost_adv, df_adv= run_advanced_lp_sim(
                    daily_house2, daily_solar2, daily_ev2,
                    ev_arr, ev_dep,
                    battery_capacity= unified_batt_capacity,
                    ev_batt_capacity= 50,
                    demand_charge_enabled= en_demand,
                    battery_mode= adv_mode,
                    backup_reserve_frac= backup_r
                )
                if cost_adv is not None:
                    st.success(f"Advanced LP Annual Cost= ${cost_adv:,.2f}")
                    st.write(df_adv.head(20))
                else:
                    st.warning("No solution or error in solver.")

    with tab4:
        st.header("Battery Size Sweep")
        if not PULP_AVAILABLE:
            st.error("No PuLP => can't do param sweep.")
        else:
            st.write("Select battery sizes to test (0..20, step 2 for instance).")
            if st.button("Run Param Sweep"):
                daily_house_sw= np.full(DAYS_PER_YEAR, house_kwh_base*(1+fluct))
                daily_solar_sw= np.full(DAYS_PER_YEAR, solar_size*4)
                daily_ev_sw= np.full(DAYS_PER_YEAR, (commute_miles/4.0)*(days_per_week/7.0))

                if monthly_charge_time=="Night (Super Off-Peak)":
                    ev_arr_s=18; ev_dep_s=7
                else:
                    ev_arr_s=9; ev_dep_s=16

                sizes= np.arange(0,21,2)
                df_sweep= param_sweep_battery_sizes(
                    daily_house_sw, daily_solar_sw, daily_ev_sw,
                    sizes, ev_arr_s, ev_dep_s,
                    ev_batt_capacity=50,
                    demand_charge_enabled=False, # or a checkbox
                    battery_mode="None"
                )
                st.dataframe(df_sweep.style.format(precision=2))
                figS, axS= plt.subplots()
                axS.plot(df_sweep["BatterySize(kWh)"], df_sweep["AnnualCost($)"], marker="o")
                axS.set_xlabel("BatterySize(kWh)")
                axS.set_ylabel("AnnualCost($)")
                st.pyplot(figS)

    with tab5:
        st.header("Recommendations (deepseek R1)")

        st.write("If you have an uploaded PDF bill, we can generate custom suggestions.")
        if st.button("Generate Recommendations"):
            recs = get_deepseek_recommendations(user_bill_data)
            if not recs:
                st.info("No specific recommendations or missing PDF data.")
            else:
                st.success("Here are your custom tips:")
                for r in recs:
                    st.write("-", r)


if __name__=="__main__":
    main()
