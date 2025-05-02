# services/urdb_api.py
import streamlit as st
import requests

# services/urdb_api.py

import streamlit as st
import requests

def fetch_urdb_plans_for_state(state):
    api_key = st.secrets["URDB_API_KEY"]
    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 8,
        "format": "json",
        "api_key": api_key,
        "country": "USA",
        "state": state,
        "detail": "full",
        "limit": 100  # increase to get a broader set
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.warning(f"URDB request failed with status {resp.status_code}")
        return []

    data = resp.json()
    if "items" not in data:
        return []

    raw_plans = data["items"]

    # ✅ Filter: Only active, residential plans
    residential_active_plans = [
        p for p in raw_plans
        if p.get("sector") == "Residential" and not p.get("recovered", False)
    ]
    print(f"Found {len(residential_active_plans)} active residential plans")
    print(f"First plan: {residential_active_plans[0]}")
    print(f"Last plan: {residential_active_plans[-1]}")
    return residential_active_plans


def naive_parse_urdb_plan(plan_json):
    """
    This is a placeholder. Real parsing might be more complex.
    """
    return {
        "on_peak": 0.40,
        "off_peak": 0.20,
        "super_off_peak": 0.10,
        "demand_rate": 10.0
    }
