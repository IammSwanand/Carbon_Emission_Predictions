"""
Step-by-Step MLflow Learning Guide
----------------------------------
Model 1: Ridge Regression (Linear Baseline)

In this script, you will learn the core MLflow primitives:
1. mlflow.set_experiment()   -> Organizes your work into a named project container.
2. mlflow.start_run()        -> Starts a tracked execution session (records params, metrics, model).
3. mlflow.log_param()        -> Logs configuration choices (e.g., alpha, scaler, features).
4. mlflow.log_metric()       -> Logs numerical evaluation results (e.g., R2, RMSE).
5. mlflow.sklearn.log_model()-> Packages and saves the model pipeline with input/output signature.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.config import (
    SELECTED_FEATURES,
    TARGET_COL,
    COUNTRY_COL,
    MODEL_ARTIFACT_PATH,
    MLFLOW_EXPERIMENT_NAME,
)
from src.data import load_dataset, get_group_kfold_splits, get_temporal_split


def train_single_model():
    # ---------------------------------------------------------
    # STEP 1: Set the MLflow Experiment
    # ---------------------------------------------------------
    # An 'Experiment' in MLflow is like a project folder for related runs.
    # If the experiment doesn't exist, MLflow creates it automatically.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"[EXPERIMENT] Active MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")

    # ---------------------------------------------------------
    # STEP 2: Load the Dataset
    # ---------------------------------------------------------
    df = load_dataset()
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL].values
    print(f"[DATA] Loaded {len(df)} rows across {df[COUNTRY_COL].nunique()} countries.")

    # ---------------------------------------------------------
    # STEP 3: Define the Model Pipeline
    # ---------------------------------------------------------
    # Ridge regression requires standardized features because it penalizes large weights.
    alpha_value = 1.0
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha_value, random_state=42)),
    ])

    # ---------------------------------------------------------
    # STEP 4: Start an MLflow Run
    # ---------------------------------------------------------
    # 'with mlflow.start_run()' opens a tracking context.
    # Everything logged inside this block belongs to this specific run.
    with mlflow.start_run(run_name="01_Ridge_Baseline"):
        print("\n[RUN] Started MLflow Run: '01_Ridge_Baseline'")

        # A. Log Parameters (Hyperparameters & Setup)
        mlflow.log_param("model_type", "Ridge Regression")
        mlflow.log_param("alpha", alpha_value)
        mlflow.log_param("features", SELECTED_FEATURES)
        mlflow.log_param("num_features", len(SELECTED_FEATURES))
        print("   -> Logged parameters: model_type, alpha, feature list")

        # B. Evaluate using GroupKFold (Spatial generalization - unseen countries)
        group_splits = list(get_group_kfold_splits(df, n_splits=5))
        gkf_r2_list, gkf_rmse_list = [], []

        for train_idx, val_idx in group_splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)

            gkf_r2_list.append(r2_score(y_val, preds))
            gkf_rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))

        mean_gkf_r2 = float(np.mean(gkf_r2_list))
        mean_gkf_rmse = float(np.mean(gkf_rmse_list))

        # C. Evaluate using Out-of-Time Temporal Split (Train <= 2003, Test 2004-2008)
        train_temporal_df, test_temporal_df = get_temporal_split(df, split_year=2003)
        pipeline.fit(train_temporal_df[SELECTED_FEATURES], train_temporal_df[TARGET_COL].values)
        temporal_preds = pipeline.predict(test_temporal_df[SELECTED_FEATURES])

        temporal_r2 = float(r2_score(test_temporal_df[TARGET_COL].values, temporal_preds))
        temporal_rmse = float(np.sqrt(mean_squared_error(test_temporal_df[TARGET_COL].values, temporal_preds)))
        temporal_mae = float(mean_absolute_error(test_temporal_df[TARGET_COL].values, temporal_preds))

        # D. Log Metrics to MLflow
        mlflow.log_metric("group_cv_r2", mean_gkf_r2)
        mlflow.log_metric("group_cv_rmse", mean_gkf_rmse)
        mlflow.log_metric("temporal_r2", temporal_r2)
        mlflow.log_metric("temporal_rmse", temporal_rmse)
        mlflow.log_metric("temporal_mae", temporal_mae)
        print(f"   -> Logged metrics: GroupKFold R2={mean_gkf_r2:.3f} | Temporal R2={temporal_r2:.3f}")

        # E. Fit Final Pipeline on Full Dataset
        pipeline.fit(X, y)

        # F. Infer Signature & Log the Model Artifact
        input_sample = X.iloc[:5]
        signature = infer_signature(input_sample, pipeline.predict(input_sample))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_sample.iloc[:1],
            serialization_format="cloudpickle",
        )
        print("   -> Logged trained Scikit-Learn pipeline artifact to MLflow")

        # G. Save local production copy for FastAPI / Streamlit
        joblib.dump(pipeline, MODEL_ARTIFACT_PATH, compress=3)
        print(f"   -> Saved local pipeline artifact to {MODEL_ARTIFACT_PATH}")

    print("\n[SUCCESS] Run completed successfully! You can view it in the MLflow UI.")


if __name__ == "__main__":
    train_single_model()
