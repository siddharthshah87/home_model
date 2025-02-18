# models/monthly_model.py
import numpy as np

DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]
SUMMER_MONTHS = [5,6,7,8]
WINTER_MONTHS = [0,1,2,3,4,9,10,11]

def calculate_monthly_values(daily_value):
    return [daily_value*d for d in DAYS_IN_MONTH]

def calculate_ev_demand(miles, efficiency, days_per_week=7):
    daily_demand = miles/ efficiency
    total_days = days_per_week*52
    yearly= daily_demand * total_days
    monthly= calculate_monthly_values(daily_demand*(days_per_week/7.0))
    return yearly, monthly

def calculate_solar_production(size_kw):
    yearly= size_kw*4*365
    monthly= calculate_monthly_values(size_kw*4)
    return yearly, monthly

def calculate_monthly_costs(ev_monthly, solar_monthly, house_monthly,
                            battery_capacity, time_of_charging):
    ev_no=[]
    ev_n2=[]
    ev_n3=[]
    tot_no=[]
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

        # (1) No Solar
        house_cost_ns= house_monthly[m]* off_peak
        ev_cost_ns= ev_monthly[m]* super_off
        ev_no.append(ev_cost_ns)
        tot_no.append(house_cost_ns+ ev_cost_ns)

        # (2) NEM 2.0
        leftover_solar= max(0, solar_monthly[m]- house_monthly[m])
        house_cost_n2= house_monthly[m]* off_peak
        ev_kwh= ev_monthly[m]

        if time_of_charging=="Night (Super Off-Peak)":
            ev_rate= super_off
        else:
            ev_rate= on_peak

        if leftover_solar>= ev_kwh:
            leftover_export= leftover_solar- ev_kwh
            ev_cost2= 0.0
            credit_n2= leftover_export* off_peak
        else:
            offset_kwh= leftover_solar
            leftover_export=0
            ev_grid_kwh= ev_kwh- offset_kwh
            ev_cost2= ev_grid_kwh* ev_rate
            credit_n2=0.0
        cost_n2= house_cost_n2+ ev_cost2 - credit_n2
        ev_n2.append(ev_cost2)
        tot_n2.append(cost_n2)

        # (3) NEM 3 + naive battery
        house_cost_3= house_monthly[m]* off_peak
        leftover_sol_3= max(0, solar_monthly[m]- house_monthly[m])
        ev_short= ev_monthly[m]

        if time_of_charging=="Daytime (Peak)" and leftover_sol_3>0:
            direct= min(ev_short, leftover_sol_3)
            ev_short-= direct
            leftover_sol_3-= direct

        # battery naive usage
        if leftover_sol_3>0 and battery_state< battery_capacity:
            can_chg= min(leftover_sol_3, battery_capacity- battery_state)
            battery_state+= can_chg*0.9
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

    return (ev_no, ev_n2, ev_n3, tot_no, tot_n2, tot_n3)
