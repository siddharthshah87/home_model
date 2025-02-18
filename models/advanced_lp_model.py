# models/advanced_lp_model.py
import numpy as np
import pandas as pd

try:
    import pulp
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

def generate_utility_rate_schedule():
    data=[]
    for m in range(12):
        for d_type in ["weekday","weekend"]:
            for h in range(24):
                if 16<=h<20:
                    import_r=0.45
                    export_r=0.05
                    demand_r=10.0
                elif 7<=h<16 or 20<=h<22:
                    import_r=0.25
                    export_r=0.03
                    demand_r=5.0
                else:
                    import_r=0.12
                    export_r=0.01
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

def can_charge_this_hour(h, arr, dep):
    if arr<dep:
        return arr<= h< dep
    else:
        return (h>= arr) or (h< dep)

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
    # None => batt=0 => no constraints

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
    if battery_mode=="None":
        battery_capacity=0

    prob= LpProblem(f"Day_{day_idx}_Dispatch", LpMinimize)

    home_batt_in  = pulp.LpVariable.dicts("batt_in", range(24), lowBound=0)
    home_batt_out = pulp.LpVariable.dicts("batt_out", range(24), lowBound=0)
    ev_charge     = pulp.LpVariable.dicts("ev_charge", range(24), lowBound=0)
    grid_import   = pulp.LpVariable.dicts("grid_import", range(24), lowBound=0)
    grid_export   = pulp.LpVariable.dicts("grid_export", range(24), lowBound=0)
    soc= [pulp.LpVariable(f"soc_{h}", lowBound=0, upBound=battery_capacity) for h in range(25)]

    peak_demand= pulp.LpVariable("peak_demand", lowBound=0)
    cost_import=[]
    credit_export=[]

    for h in range(24):
        import_r= df_rates_day.loc[h,"import_rate"]
        export_r= df_rates_day.loc[h,"export_rate"]

        if not can_charge_this_hour(h, ev_arr, ev_dep):
            prob += ev_charge[h]==0

        prob += (solar_24[h] + home_batt_out[h] + grid_import[h]
                 == house_24[h] + ev_charge[h] + home_batt_in[h] + grid_export[h])
        prob += soc[h+1]== soc[h] + home_batt_in[h] - home_batt_out[h]

        if demand_charge_enabled:
            prob += peak_demand>= grid_import[h]

        cost_import.append(grid_import[h]* import_r)
        credit_export.append(grid_export[h]* export_r)

        advanced_battery_constraints(prob,h,h,
                                     battery_mode, backup_reserve_frac,
                                     home_batt_out, grid_export, soc,
                                     battery_capacity,
                                     self_consumption_excess)

    # EV daily total
    prob += sum(ev_charge[h] for h in range(24))== ev_kwh_day
    prob += soc[0]== start_batt_soc
    prob += soc[24]== soc[0]
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

def run_advanced_lp_sim(daily_house,
                        daily_solar,
                        daily_ev,
                        ev_arr=18,
                        ev_dep=7,
                        battery_capacity=10,
                        ev_batt_capacity=50,
                        demand_charge_enabled=False,
                        battery_mode="None",
                        backup_reserve_frac=0.2,
                        self_consumption_excess="Curtail"):
    df_rates= generate_utility_rate_schedule()
    cum_days= np.cumsum([0]+[31,28,31,30,31,30,31,31,30,31,30,31])
    total_cost=0.0
    day_solutions=[]
    battery_soc=0.0

    for day_idx in range(365):
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
        day_solutions.append({"day_idx":day_idx,"cost_day": day_cost if day_cost else 0})

    df_sol= pd.DataFrame(day_solutions)
    return total_cost, df_sol
