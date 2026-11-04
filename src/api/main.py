import json
import logging
from typing import Dict, Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    MODEL_ARTIFACT_PATH,
    SELECTED_FEATURES,
    MODELS_DIR,
)
from src.api.schemas import (
    CountryFeaturesInput,
    PredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    ScenarioProjectionInput,
    ScenarioProjectionResponse,
    HealthResponse,
)

logger = logging.getLogger("carbon_api")
logging.basicConfig(level=logging.INFO)

state: Dict[str, Any] = {
    "pipeline": None,
    "feature_bounds": None,
    "version": "1.0.0",
    "model_name": "VotingEnsemble (Ridge + RF + LightGBM)",
}


def load_artifacts():
    """Load model pipeline and bounds safely."""
    if state["pipeline"] is None and MODEL_ARTIFACT_PATH.exists():
        try:
            state["pipeline"] = joblib.load(MODEL_ARTIFACT_PATH)
            logger.info("Successfully loaded ML pipeline from %s", MODEL_ARTIFACT_PATH)
        except Exception as e:
            logger.error("Error loading model pipeline: %s", e)

    dist_path = MODELS_DIR / "feature_distribution.json"
    if state["feature_bounds"] is None and dist_path.exists():
        try:
            with open(dist_path, "r") as f:
                state["feature_bounds"] = json.load(f)
            logger.info("Loaded feature distribution bounds.")
        except Exception as e:
            logger.error("Error loading feature bounds: %s", e)


# Eager load artifacts
load_artifacts()

app = FastAPI(
    title="Global CO2 Emissions Prediction API",
    description=(
        "Production REST API serving trained climate regression models. "
        "Includes Pydantic validation, Out-Of-Distribution (OOD) detection, "
        "and policy scenario projection simulation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_ood(features_dict: Dict[str, float]) -> tuple[bool, List[str]]:
    """Check whether feature values exceed historical bounds (min/max from training)."""
    bounds = state.get("feature_bounds")
    if not bounds:
        return False, []

    is_ood = False
    warnings = []
    for feat in SELECTED_FEATURES:
        if feat in bounds and feat in features_dict:
            f_min = bounds[feat]["min"]
            f_max = bounds[feat]["max"]
            val = features_dict[feat]
            if val < f_min or val > f_max:
                is_ood = True
                warnings.append(
                    f"Feature '{feat}' value {val:.2f} is outside historical training range [{f_min:.2f}, {f_max:.2f}]."
                )

    return is_ood, warnings


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Global CO2 Emissions Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": state["version"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    load_artifacts()
    return HealthResponse(
        status="healthy" if state["pipeline"] is not None else "degraded",
        model_loaded=state["pipeline"] is not None,
        features=SELECTED_FEATURES,
        version=state["version"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(payload: CountryFeaturesInput):
    load_artifacts()
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training pipeline has been executed.",
        )

    data_dict = payload.model_dump()
    is_ood, warnings = check_ood(data_dict)

    # Format into DataFrame for Scikit-Learn pipeline
    df_input = pd.DataFrame([data_dict])[SELECTED_FEATURES]

    try:
        prediction = float(pipeline.predict(df_input)[0])
        prediction = max(0.0, prediction)
    except Exception as e:
        logger.error("Inference failure: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return PredictionResponse(
        predicted_co2_per_cap=round(prediction, 4),
        model_name=state["model_name"],
        model_version=state["version"],
        is_out_of_distribution=is_ood,
        warnings=warnings,
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictionInput):
    load_artifacts()
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    rows = [item.model_dump() for item in payload.records]
    df_input = pd.DataFrame(rows)[SELECTED_FEATURES]

    try:
        preds = pipeline.predict(df_input)
        preds = [round(max(0.0, float(p)), 4) for p in preds]
    except Exception as e:
        logger.error("Batch inference failure: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return BatchPredictionResponse(
        predictions=preds,
        count=len(preds),
        model_version=state["version"],
    )


@app.post("/scenario", response_model=ScenarioProjectionResponse, tags=["Simulation"])
def simulate_scenario_endpoint(payload: ScenarioProjectionInput):
    load_artifacts()
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    current = payload.base_features.model_dump()
    timeline = []
    projected_co2 = []
    growth_rates = payload.growth_rate_overrides or {}

    for year_offset in range(1, payload.years_ahead + 1):
        year = payload.start_year + year_offset
        for feat in SELECTED_FEATURES:
            rate = growth_rates.get(feat, 0.0)
            current[feat] = current[feat] * (1.0 + rate)

        df_row = pd.DataFrame([current])[SELECTED_FEATURES]
        pred = max(0.0, float(pipeline.predict(df_row)[0]))

        timeline.append(year)
        projected_co2.append(round(pred, 4))

    return ScenarioProjectionResponse(
        country=payload.country,
        timeline=timeline,
        projected_co2=projected_co2,
        assumed_growth_rates=growth_rates,
    )
