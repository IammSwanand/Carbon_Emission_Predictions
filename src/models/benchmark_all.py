import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import (
    SELECTED_FEATURES,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    RANDOM_STATE,
    N_SPLITS_CV,
    MODEL_ARTIFACT_PATH,
    MODELS_DIR,
    METRICS_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
)
from src.data import (
    load_dataset,
    get_kfold_splits,
    get_group_kfold_splits,
    get_temporal_split,
)


def get_base_sub_pipelines():
    """Create sub-pipelines for base estimators in ensembles."""
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    rf_est = RandomForestRegressor(
        n_estimators=200,
        max_depth=16,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_est = LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    return [("ridge", ridge_pipe), ("rf", rf_est), ("lgb", lgb_est)]


def get_model_candidates() -> Dict[str, Any]:
    """Define single models and meta-ensembles."""
    base_sub = get_base_sub_pipelines()

    candidates = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        ),
        "VotingEnsemble": VotingRegressor(
            estimators=base_sub,
            weights=[1, 2, 2],
            n_jobs=-1,
        ),
        "StackingEnsemble": StackingRegressor(
            estimators=base_sub,
            final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
            cv=3,
            n_jobs=-1,
        ),
    }
    return candidates


def build_pipeline(estimator_name: str, estimator: Any) -> Pipeline:
    """Wrap estimator in a Pipeline with column selection and scaling if needed."""
    if estimator_name == "Ridge":
        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), SELECTED_FEATURES)],
            remainder="drop",
        )
    else:
        # Ensembles handle their own internal scalers or are tree-based
        preprocessor = ColumnTransformer(
            transformers=[("pass", "passthrough", SELECTED_FEATURES)],
            remainder="drop",
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])


def evaluate_cv(
    pipeline: Pipeline, df: pd.DataFrame, cv_splits: List[tuple]
) -> Dict[str, float]:
    """Evaluate pipeline across CV folds and return mean and std metrics."""
    r2_scores, rmse_scores, mae_scores = [], [], []

    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL].values

    for train_idx, val_idx in cv_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        r2_scores.append(r2_score(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Execute evaluation across Random CV, GroupKFold (Country), and Temporal Split."""
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset loaded: {df.shape[0]} rows, {df[COUNTRY_COL].nunique()} countries.")

    # Prepare CV splits
    random_splits = list(get_kfold_splits(df, n_splits=N_SPLITS_CV))
    group_splits = list(get_group_kfold_splits(df, n_splits=N_SPLITS_CV))
    train_temporal_df, test_temporal_df = get_temporal_split(df)

    candidates = get_model_candidates()
    results = {}

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_score = -1.0
    best_model_name = None

    for name, estimator in candidates.items():
        print(f"\n--- Benchmarking {name} ---")
        pipeline = build_pipeline(name, estimator)

        # 1. Naive Random K-Fold (Data Leakage)
        random_cv_metrics = evaluate_cv(pipeline, df, random_splits)

        # 2. GroupKFold by Country (Spatial generalization)
        group_cv_metrics = evaluate_cv(pipeline, df, group_splits)

        # 3. Temporal Out-of-Time Split (Train <= 2003, Test 2004-2008)
        X_train_temp = train_temporal_df[SELECTED_FEATURES]
        y_train_temp = train_temporal_df[TARGET_COL].values
        X_test_temp = test_temporal_df[SELECTED_FEATURES]
        y_test_temp = test_temporal_df[TARGET_COL].values

        pipeline.fit(X_train_temp, y_train_temp)
        start_time = time.time()
        temp_preds = pipeline.predict(X_test_temp)
        latency_ms = ((time.time() - start_time) / len(X_test_temp)) * 1000

        temporal_metrics = {
            "r2": float(r2_score(y_test_temp, temp_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_temp, temp_preds))),
            "mae": float(mean_absolute_error(y_test_temp, temp_preds)),
            "latency_ms_per_sample": float(latency_ms),
        }

        results[name] = {
            "random_cv": random_cv_metrics,
            "group_cv_country": group_cv_metrics,
            "temporal_split": temporal_metrics,
        }

        # Track best model by GroupKFold R2 (leakage-free metric)
        if group_cv_metrics["r2_mean"] > best_score:
            best_score = group_cv_metrics["r2_mean"]
            best_model_name = name

        # Log run to MLflow
        with mlflow.start_run(run_name=f"{name}_benchmark"):
            mlflow.log_param("model_name", name)
            mlflow.log_param("is_ensemble", "Ensemble" in name)
            mlflow.log_param("features", SELECTED_FEATURES)
            
            # Metrics
            mlflow.log_metric("random_cv_r2_mean", random_cv_metrics["r2_mean"])
            mlflow.log_metric("random_cv_rmse_mean", random_cv_metrics["rmse_mean"])
            mlflow.log_metric("group_cv_r2_mean", group_cv_metrics["r2_mean"])
            mlflow.log_metric("group_cv_rmse_mean", group_cv_metrics["rmse_mean"])
            mlflow.log_metric("temporal_r2", temporal_metrics["r2"])
            mlflow.log_metric("temporal_rmse", temporal_metrics["rmse"])
            mlflow.log_metric("latency_ms", temporal_metrics["latency_ms_per_sample"])

            # Infer model signature & log model artifact using cloudpickle
            input_sample = df[SELECTED_FEATURES].iloc[:5]
            pred_sample = pipeline.predict(input_sample)
            signature = infer_signature(input_sample, pred_sample)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=input_sample.iloc[:1],
                serialization_format="cloudpickle",
            )

        print(
            f"Random CV R2: {random_cv_metrics['r2_mean']:.3f} | "
            f"GroupKFold R2: {group_cv_metrics['r2_mean']:.3f} | "
            f"Temporal R2: {temporal_metrics['r2']:.3f}"
        )

    # Train best model on full dataset for production artifact
    print(f"\nWinner: {best_model_name} (GroupKFold R2 = {best_score:.3f})")
    print(f"Training production pipeline using {best_model_name} on entire dataset...")
    best_estimator = candidates[best_model_name]
    prod_pipeline = build_pipeline(best_model_name, best_estimator)
    prod_pipeline.fit(df[SELECTED_FEATURES], df[TARGET_COL].values)

    joblib.dump(prod_pipeline, MODEL_ARTIFACT_PATH, compress=3)
    print(f"Production pipeline saved to {MODEL_ARTIFACT_PATH}")

    # Log champion model to MLflow Model Registry
    with mlflow.start_run(run_name="Champion_Production_Pipeline"):
        mlflow.log_param("champion_model", best_model_name)
        mlflow.log_metric("champion_group_cv_r2", best_score)
        input_sample = df[SELECTED_FEATURES].iloc[:5]
        signature = infer_signature(input_sample, prod_pipeline.predict(input_sample))
        mlflow.sklearn.log_model(
            sk_model=prod_pipeline,
            name="production_model",
            signature=signature,
            registered_model_name="CarbonEmissionsChampion",
            serialization_format="cloudpickle",
        )
        # Log artifacts (figures and metrics)
        for fig_file in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(fig_file), artifact_path="figures")

    # Save feature distributions for out-of-distribution (OOD) checks
    stats = {}
    for col in SELECTED_FEATURES:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }
    with open(MODELS_DIR / "feature_distribution.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save benchmark summary table
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved to {METRICS_PATH}")
    return results


if __name__ == "__main__":
    run_benchmarks()
