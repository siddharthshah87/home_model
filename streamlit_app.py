import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Constants
TOU_RATES = {
    "summer": {"on_peak": 0.45, "off_peak": 0.25, "super_off_peak": 0.12},
    "winter": {"on_peak": 0.35, "off_peak": 0.20, "super_off_peak": 0.10},
}
SUMMER_MONTHS = [5, 6, 7, 8]
WINTER_MONTHS = [0, 1, 2, 3, 4, 9, 10, 11]
DEFAULT_BATTERY_EFFICIENCY = 0.9  # 90% efficiency

# Core function to calculate monthly costs
def calculate_monthly_costs_updated(solar, household, ev, battery_capacity, charging_time):
    """Updated monthly cost calculation logic with detailed battery usage breakdown."""
    ev_cost_no_solar = []
    ev_cost_nem_2 = []
    nem_3_battery_costs = []
    total_cost_no_solar = []
    total_cost_nem_2 = []
    total_cost_nem_3 = []
    battery_usage = []  # Track how much energy is supplied by the battery

    battery_state = 0  # Start with an empty battery

    for month in range(12):
        rates = TOU_RATES["summer"] if month in SUMMER_MONTHS else TOU_RATES["winter"]

        # Household costs (no solar)
        household_cost_no_solar = household * rates["off_peak"]

        # EV cost (no solar, charged at night)
        ev_cost = ev * rates["super_off_peak"]
        ev_cost_no_solar.append(ev_cost)

        # Total cost without solar
        total_cost_no_solar.append(household_cost_no_solar + ev_cost)

        # EV cost under NEM 2.0
        if solar > 0:
            excess_solar = solar - household
            credit_nem_2 = max(0, excess_solar * rates["off_peak"])
            ev_cost_under_nem_2 = max(0, ev - credit_nem_2)
        else:
            credit_nem_2 = 0  # No credits when solar is zero
            ev_cost_under_nem_2 = ev * rates["super_off_peak"]  # All grid-powered EV charging

        ev_cost_nem_2.append(ev_cost_under_nem_2)
        total_cost_nem_2.append(household_cost_no_solar - credit_nem_2 + ev_cost_under_nem_2)

        # EV cost under NEM 3.0 + Battery
        excess_solar = max(0, solar - household)

        # Charge battery with remaining solar
        if excess_solar > 0 and battery_state < battery_capacity:
            battery_charge = min(excess_solar, battery_capacity - battery_state)
            battery_state += battery_charge * DEFAULT_BATTERY_EFFICIENCY
            excess_solar -= battery_charge

        # Discharge battery to meet EV demand (Nighttime or later charging priority)
        ev_shortfall = ev
        battery_contribution = 0

        if battery_state > 0:
            battery_discharge = min(ev_shortfall, battery_state)
            ev_shortfall -= battery_discharge
            battery_state -= battery_discharge
            battery_contribution += battery_discharge

        battery_usage.append(battery_contribution)

        # Remaining EV demand is met by the grid
        if charging_time == "Night (Super Off-Peak)":
            grid_energy_used = ev_shortfall
            nem_3_cost = grid_energy_used * rates["super_off_peak"]
        elif charging_time == "Daytime (Peak)":
            grid_energy_used = ev_shortfall
            nem_3_cost = grid_energy_used * rates["on_peak"]

        nem_3_battery_costs.append(nem_3_cost)
        total_cost_nem_3.append(household_cost_no_solar + nem_3_cost)

    return ev_cost_no_solar, ev_cost_nem_2, nem_3_battery_costs, total_cost_no_solar, total_cost_nem_2, total_cost_nem_3, battery_usage

# Define simulation parameters for visualizations
def simulate_scenarios():
    scenarios = [
        {"solar": 1000, "household": 400, "ev": 100, "battery": 10, "charging_time": "Night (Super Off-Peak)"},
        {"solar": 300, "household": 400, "ev": 200, "battery": 10, "charging_time": "Daytime (Peak)"},
        {"solar": 0, "household": 400, "ev": 200, "battery": 0, "charging_time": "Night (Super Off-Peak)"},
    ]

    results = []

    for scenario in scenarios:
        ev_cost_no_solar, ev_cost_nem_2, nem_3_battery_costs, total_cost_no_solar, total_cost_nem_2, total_cost_nem_3, battery_usage = calculate_monthly_costs_updated(
            scenario["solar"],
            scenario["household"],
            scenario["ev"],
            scenario["battery"],
            scenario["charging_time"],
        )
        results.append({
            "Scenario": scenario,
            "EV Cost (No Solar)": sum(ev_cost_no_solar),
            "EV Cost (NEM 2.0)": sum(ev_cost_nem_2),
            "EV Cost (NEM 3.0 + Battery)": sum(nem_3_battery_costs),
            "Total Cost (No Solar)": sum(total_cost_no_solar),
            "Total Cost (NEM 2.0)": sum(total_cost_nem_2),
            "Total Cost (NEM 3.0 + Battery)": sum(total_cost_nem_3),
            "Battery Contribution (kWh)": sum(battery_usage),
        })

    return results

# Visualization and analysis
def create_visualizations(results):
    st.title("Energy Simulation Dashboard")

    # Create tabs for interactive adjustments and scenarios
    tabs = st.tabs(["Interactive Simulation", "Predefined Scenarios", "Comparison"])

    # Tab 1: Interactive Simulation
    with tabs[0]:
        st.sidebar.header("Adjust Simulation Parameters")

        solar = st.sidebar.slider("Solar Production (kWh per month)", 0, 1500, 1000, 50)
        household = st.sidebar.slider("Household Consumption (kWh per month)", 0, 1000, 400, 50)
        ev = st.sidebar.slider("EV Consumption (kWh per month)", 0, 500, 100, 10)
        battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 0, 20, 10, 1)
        charging_time = st.sidebar.radio("EV Charging Time", ["Night (Super Off-Peak)", "Daytime (Peak)"])

        ev_cost_no_solar, ev_cost_nem_2, nem_3_battery_costs, total_cost_no_solar, total_cost_nem_2, total_cost_nem_3, battery_usage = calculate_monthly_costs_updated(
            solar, household, ev, battery_capacity, charging_time
        )

        st.header("Live Simulation Results")
        results_df = pd.DataFrame({
            "Metric": ["EV Cost (No Solar)", "EV Cost (NEM 2.0)", "EV Cost (NEM 3.0 + Battery)", "Total Cost (No Solar)", "Total Cost (NEM 2.0)", "Total Cost (NEM 3.0 + Battery)", "Battery Contribution (kWh)"],
            "Monthly Value": [
                sum(ev_cost_no_solar) / 12,
                sum(ev_cost_nem_2) / 12,
                sum(nem_3_battery_costs) / 12,
                sum(total_cost_no_solar) / 12,
                sum(total_cost_nem_2) / 12,
                sum(total_cost_nem_3) / 12,
                sum(battery_usage) / 12,
            ],
            "Yearly Value": [
                sum(ev_cost_no_solar),
                sum(ev_cost_nem_2),
                sum(nem_3_battery_costs),
                sum(total_cost_no_solar),
                sum(total_cost_nem_2),
                sum(total_cost_nem_3),
                sum(battery_usage),
            ],
        })

        st.table(results_df)

    # Tab 2: Predefined Scenarios
    with tabs[1]:
        st.header("Predefined Scenarios")
        for idx, result in enumerate(results):
            st.subheader(f"Scenario {idx + 1} Results")

            scenario = result["Scenario"]
            st.write("**Scenario Parameters:**")
            st.json(scenario)

            st.write("**Results Summary:**")
            summary_df = pd.DataFrame({
                "Metric": ["EV Cost (No Solar)", "EV Cost (NEM 2.0)", "EV Cost (NEM 3.0 + Battery)", "Total Cost (No Solar)", "Total Cost (NEM 2.0)", "Total Cost (NEM 3.0 + Battery)", "Battery Contribution (kWh)"],
                "Value": [
                    result["EV Cost (No Solar)"],
                    result["EV Cost (NEM 2.0)"],
                    result["EV Cost (NEM 3.0 + Battery)"],
                    result["Total Cost (No Solar)"],
                    result["Total Cost (NEM 2.0)"],
                    result["Total Cost (NEM 3.0 + Battery)"],
                    result["Battery Contribution (kWh)"],
                ]
            })
            st.table(summary_df)

    # Tab 3: Comparison
    with tabs[2]:
        st.header("Comparison of All Scenarios")
        comparison_df = pd.DataFrame(results)

        for col in comparison_df.columns:
            if col != "Scenario":
                comparison_df[col] = comparison_df[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        st.dataframe(comparison_df)

        # Visualization for comparison
        st.write("### Cost and Battery Contribution Comparison")
        fig, ax = plt.subplots()
        metrics = ["EV Cost (No Solar)", "EV Cost (NEM 2.0)", "EV Cost (NEM 3.0 + Battery)", "Total Cost (No Solar)", "Total Cost (NEM 2.0)", "Total Cost (NEM 3.0 + Battery)"]
        for metric in metrics:
            ax.plot([f"Scenario {i+1}" for i in range(len(results))], [result[metric] for result in results], label=metric)

        ax.set_ylabel("Cost ($)")
        ax.set_title("Cost Comparison")
        ax.legend()
        st.pyplot(fig)

# Run simulation and display visualizations
results = simulate_scenarios()
create_visualizations(results)
