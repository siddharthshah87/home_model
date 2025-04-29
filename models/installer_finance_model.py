# models/installer_finance_model.py (New)

def estimate_payback_costs(panel_type, smart_features=True,
                            panel_upgrade_avoided=True,
                            v2h_enabled=True,
                            base_cost=2500,
                            install_cost=1200,
                            panel_upgrade_cost=4000):
    """
    Simple financial model to estimate effective system cost and payback.
    """
    total_cost = base_cost + install_cost

    if panel_upgrade_avoided:
        total_cost -= panel_upgrade_cost

    if not smart_features:
        total_cost -= 500  # no load control savings

    if not v2h_enabled:
        total_cost -= 750  # no inverter/V2H interface

    return max(total_cost, 0)
