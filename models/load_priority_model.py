# models/load_priority_model.py

CRITICAL_LOADS = ["refrigerator", "lighting", "wifi", "garage_opener"]
FLEXIBLE_LOADS = ["EV", "HVAC", "WasherDryer", "WaterHeater"]

def classify_load_priority(load_name):
    if load_name in CRITICAL_LOADS:
        return "critical"
    elif load_name in FLEXIBLE_LOADS:
        return "flexible"
    return "unknown"

def summarize_load_portfolio(smart_panel):
    active = smart_panel.get_controlled_loads()
    crit = [l for l in active if classify_load_priority(l) == "critical"]
    flex = [l for l in active if classify_load_priority(l) == "flexible"]
    return {
        "critical": crit,
        "flexible": flex,
        "total_managed": len(active)
    }
