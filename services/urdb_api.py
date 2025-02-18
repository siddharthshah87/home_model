# services/urdb_api.py
import streamlit as st
import requests

def fetch_urdb_plans_for_state(state):
    api_key= st.secrets["URDB_API_KEY"]
    url= "https://api.openei.org/utility_rates"
    params= {
        "version": 8,
        "format": "json",
        "api_key": api_key,
        "country": "USA",
        "state": state,
        "detail": "full",
        "limit": 50
    }
    resp= requests.get(url, params=params)
    if resp.status_code!=200:
        st.warning(f"URDB request code= {resp.status_code}")
        return []
    data= resp.json()
    if "items" not in data:
        return []
    return data["items"]

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
