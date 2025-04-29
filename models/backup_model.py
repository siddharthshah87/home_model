# models/backup_model.py (New)

import numpy as np

# Simulate outage backup support based on energy reserves and critical load

def simulate_outage_resilience(ev_battery_kwh, house_battery_kwh,
                               critical_load_kw=2.5,
                               avg_outage_hours=4,
                               outages_per_year=3,
                               smart_panel=None):
    """
    Simulates if energy system can handle typical outage events.
    Returns:
        total_covered_hours: estimated number of outage hours backed up
        coverage_rate: % of outage hours that can be served
    """
    total_outage_hours = avg_outage_hours * outages_per_year
    
    # Determine usable battery energy
    usable_ev_kwh = ev_battery_kwh * 0.9 if smart_panel and smart_panel.is_load_controlled("ev") else 0.0
    usable_house_kwh = house_battery_kwh * 0.9 if smart_panel and smart_panel.is_load_controlled("battery") else 0.0

    total_kwh_available = usable_ev_kwh + usable_house_kwh

    # Calculate how many hours this can sustain the critical load
    hours_supported = total_kwh_available / critical_load_kw if critical_load_kw > 0 else 0.0
    
    coverage_rate = min(1.0, hours_supported / total_outage_hours)

    return {
        "total_outage_hours": total_outage_hours,
        "hours_supported": round(hours_supported, 2),
        "coverage_rate": round(coverage_rate * 100, 1),  # as percentage
        "resilience_score": "✅ Excellent" if coverage_rate >= 1.0 else
                            "🟡 Partial" if coverage_rate >= 0.5 else
                            "🔴 Poor"
    }

# Example usage:
# result = simulate_outage_resilience(ev_battery_kwh=60, house_battery_kwh=10, smart_panel=my_panel)
# print(result)  # will include hours supported and resilience score
