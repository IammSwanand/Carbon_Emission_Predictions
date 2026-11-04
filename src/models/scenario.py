import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from src.config import (
    MODEL_ARTIFACT_PATH,
    SELECTED_FEATURES,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    FIGURES_DIR,
    MODELS_DIR,
)
from src.data import load_dataset


def calculate_historical_cagr(
    df: pd.DataFrame, country: str
) -> Dict[str, float]:
    """Calculate historical Compound Annual Growth Rate (CAGR) for country features."""
    c_data = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
    if c_data.empty or len(c_data) < 2:
        return {feat: 0.0 for feat in SELECTED_FEATURES}

    start_year = c_data[YEAR_COL].min()
    end_year = c_data[YEAR_COL].max()
    years = end_year - start_year

    if years <= 0:
        return {feat: 0.0 for feat in SELECTED_FEATURES}

    growth_rates = {}
    for feat in SELECTED_FEATURES:
        s_val = c_data[c_data[YEAR_COL] == start_year][feat].values[0]
        e_val = c_data[c_data[YEAR_COL] == end_year][feat].values[0]
        if s_val > 0 and e_val > 0 and np.isfinite(s_val) and np.isfinite(e_val):
            growth_rates[feat] = (e_val / s_val) ** (1.0 / years) - 1.0
        else:
            growth_rates[feat] = 0.0

    return growth_rates


def simulate_scenario(
    base_features: Dict[str, float],
    growth_rates: Dict[str, float],
    years_ahead: int = 15,
    pipeline=None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[List[float], List[bool]]:
    """
    Simulate feature evolution under growth assumptions and score against model pipeline.
    Returns:
        projected_co2: List of emissions per capita values
        is_ood_flags: List of booleans indicating whether features exited training domain
    """
    if pipeline is None:
        pipeline = joblib.load(MODEL_ARTIFACT_PATH)

    current = base_features.copy()
    projections = []
    ood_flags = []

    for _ in range(years_ahead):
        # Compound features
        for feat in SELECTED_FEATURES:
            rate = growth_rates.get(feat, 0.0)
            current[feat] = current[feat] * (1.0 + rate)

        # Domain boundary check
        ood = False
        if feature_bounds:
            for feat in SELECTED_FEATURES:
                min_val, max_val = feature_bounds[feat]
                if current[feat] < min_val or current[feat] > max_val:
                    ood = True
                    break
        ood_flags.append(ood)

        # Model inference
        df_row = pd.DataFrame([current])[SELECTED_FEATURES]
        pred = float(pipeline.predict(df_row)[0])
        projections.append(pred)

    return projections, ood_flags


def run_scenario_analysis():
    """Run baseline vs clean-transition scenario simulations for selected countries."""
    print("Running scenario simulation analysis...")
    df = load_dataset()
    pipeline = joblib.load(MODEL_ARTIFACT_PATH)

    # Load feature bounds
    bounds_path = MODELS_DIR / "feature_distribution.json"
    with open(bounds_path, "r") as f:
        dist = json.load(f)
    feature_bounds = {k: (v["min"], v["max"]) for k, v in dist.items()}

    countries = ["IND", "USA", "RUS", "NZL"]
    last_year = int(df[YEAR_COL].max())
    years_ahead = 15
    future_years = list(range(last_year + 1, last_year + 1 + years_ahead))

    results = []

    plt.figure(figsize=(12, 7))

    for country in countries:
        c_data = df[df[COUNTRY_COL] == country].sort_values(YEAR_COL)
        if c_data.empty:
            continue

        latest_row = c_data.iloc[-1]
        base_features = {f: float(latest_row[f]) for f in SELECTED_FEATURES}
        hist_cagr = calculate_historical_cagr(df, country)

        # Scenario 1: Historical CAGR continuation
        preds_base, ood_base = simulate_scenario(
            base_features, hist_cagr, years_ahead, pipeline, feature_bounds
        )

        # Scenario 2: Green Decoupling (-2% annual energy intensity, +1% conservation)
        green_rates = hist_cagr.copy()
        green_rates["en_per_cap"] = -0.015
        green_rates["prot_area_perc"] = 0.02
        preds_green, _ = simulate_scenario(
            base_features, green_rates, years_ahead, pipeline, feature_bounds
        )

        # Plot historical
        plt.plot(
            c_data[YEAR_COL],
            c_data[TARGET_COL],
            label=f"{country} (Historical)",
            linewidth=2,
        )
        # Plot projected baseline
        plt.plot(
            future_years,
            preds_base,
            "--",
            label=f"{country} (CAGR Baseline Scenario)",
            alpha=0.8,
        )
        # Plot projected green transition
        plt.plot(
            future_years,
            preds_green,
            ":",
            label=f"{country} (Green Transition Scenario)",
            alpha=0.8,
        )

        results.append({
            "country": country,
            "cagr_growth_rates": hist_cagr,
            "baseline_projections": dict(zip(future_years, preds_base)),
            "green_projections": dict(zip(future_years, preds_green)),
        })

    plt.title(
        "CO2 Emissions per Capita: Historical Trends vs Scenario Projections (2009-2023)\n"
        "(Note: Projections represent policy scenario evaluations, bounded by model domain)",
        fontsize=12,
    )
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("CO2 Emissions per Capita (metric tons)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()

    out_plot = FIGURES_DIR / "scenario_projections.png"
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Scenario plot saved to {out_plot}")
    return results


if __name__ == "__main__":
    run_scenario_analysis()

