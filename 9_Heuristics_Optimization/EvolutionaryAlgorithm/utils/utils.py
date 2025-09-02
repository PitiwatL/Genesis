import pandas as pd
import numpy as np

def simulate_clicks(assignments, propensity):
    # randomly simulate if user clicks (1) based on propensity, otw 0
    return (np.random.rand(*assignments.shape) < (propensity * assignments)).astype(int)

def update_kpi_progress(clicks, rpc, current_revenue):
    # Tang: Sum in vertical 
    return current_revenue + (clicks * rpc).sum(axis=0)

def apply_fatigue(propensity, fatigue_matrix, fatigue_weight=0.05):
    return np.clip(propensity * np.exp(-fatigue_weight * fatigue_matrix), 0, 1)

def multi_shot_simulation(
        days, 
        propensity, 
        rpc, 
        kpi_targets, 
        assign_fn, 
        user_cap=1, 
        fatigue_weight=0.05
    ):
    """
    Args:
        assign_fn: some assignment logics based on a single day.
    """
    n_users, n_products = propensity.shape
    current_revenue = np.zeros(n_products)
    fatigue_matrix = np.zeros((n_users, n_products))

    daily_assignments = []
    daily_kpis = []

    # inputs
    print("\n----------- Day 0 -----------")

    product_info = pd.DataFrame({
        "RPC": rpc,
        "KPI Target": kpi_targets
    }, index=[f"P{j}" for j in range(n_products)])
    print("\nProduct-Level Info:")
    print(product_info)

    user_propensities = pd.DataFrame(
        propensity,
        columns=[f"P{j}" for j in range(n_products)],
        index=[f"User {i+1}" for i in range(n_users)]
    )
    print("\nUser Propensities:")
    print(user_propensities)

    # simulation
    for day in range(1, days+1):
        print(f"\n----------- Day {day} -----------")

        decayed_propensity = apply_fatigue(propensity, fatigue_matrix, fatigue_weight)

        remaining_targets = kpi_targets - current_revenue
        assign_df, _, _, _ = assign_fn(
            decayed_propensity,
            rpc,
            remaining_targets,
            user_cap=user_cap
        )

        # Tang: pandas to array
        assignments = assign_df.values

        clicks = simulate_clicks(assignments, decayed_propensity)
        current_revenue = update_kpi_progress(clicks, rpc, current_revenue)

        fatigue_matrix += assignments

        kpi_achieved = current_revenue >= kpi_targets
        daily_assignments.append(assignments)
        daily_kpis.append(current_revenue >= kpi_targets)

        # for printing
        df_assign = pd.DataFrame(assignments, columns=[f"P{j}" for j in range(n_products)], index=[f"User {i+1}" for i in range(n_users)])
        df_clicks = pd.DataFrame(clicks, columns=[f"P{j}" for j in range(n_products)], index=[f"User {i+1}" for i in range(n_users)])
        df_kpi = pd.DataFrame({
            "RPC": rpc,
            "Target KPI": kpi_targets,
            "Current Rev": current_revenue.astype(int),
            "KPI Met": kpi_achieved
        }, index=[f"P{j}" for j in range(n_products)])

        print("\nAd Assignment:")
        print(df_assign)

        print("\nAd Clicks:")
        print(df_clicks)

        print("\ndecayed_propensity:")
        print(decayed_propensity)

        print("\nfatigue matrix")
        print(fatigue_matrix)

        print("\nLive KPI Status:")
        print(df_kpi)
    return daily_assignments, daily_kpis, fatigue_matrix, current_revenue
