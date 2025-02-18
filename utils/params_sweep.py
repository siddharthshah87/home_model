# utils/param_sweep.py
import numpy as np
import pandas as pd
from models.advanced_lp_model import run_advanced_lp_sim

def param_sweep_battery_sizes(daily_house,
                              daily_solar,
                              daily_ev,
                              sizes_to_test,
                              ev_arr=18,
                              ev_dep=7,
                              ev_batt_capacity=50,
                              demand_charge_enabled=False,
                              battery_mode="None",
                              backup_reserve_frac=0.2,
                              self_consumption_excess="Curtail"):
    results=[]
    for size in sizes_to_test:
        cost_sz, df_days= run_advanced_lp_sim(
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
        results.append((size, cost_sz))
    df_param= pd.DataFrame(results, columns=["BatterySize(kWh)","AnnualCost($)"])
    return df_param
