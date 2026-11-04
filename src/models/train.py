"""
Step-by-Step MLflow Learning Guide (Modular CLI Architecture)
-----------------------------------------------------------
Run specific models on demand without overwriting previous work:

Usage:
    python -m src.models.train --model ridge     # Train Model 1 (Ridge Regression)
    python -m src.models.train --model rf        # Train Model 2 (Random Forest)
    python -m src.models.train --model lgbm      # Train Model 3 (LightGBM)
    python -m src.models.train --model xgb       # Train Model 4 (XGBoost)
    python -m src.models.train --model voting    # Train Model 5 (Voting Ensemble)
    python -m src.models.train --model stacking  # Train Model 6 (Stacking Ensemble)
    python -m src.models.train --model all       # Train and log all models
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
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


def evaluate_and_log(
    pipeline: Pipeline,
    run_name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
):
    """
    Shared evaluation and MLflow logging function.
    Performs leakage-aware GroupKFold and Out-of-Time Temporal Split,
    then logs parameters, metrics, and the model artifact to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        print(f"\n[RUN] Started MLflow Run: '{run_name}'")

        # 1. Log All Parameters
        mlflow.log_params(params)
        mlflow.log_param("features", SELECTED_FEATURES)
        mlflow.log_param("num_features", len(SELECTED_FEATURES))
        print(f"   -> Logged {len(params)} hyperparameters to MLflow")

        # 2. Evaluate using GroupKFold (Spatial generalization to unseen countries)
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

        # 3. Evaluate using Out-of-Time Temporal Split (Train <= 2003, Test 2004-2008)
        train_temp_df, test_temp_df = get_temporal_split(df, split_year=2003)
        pipeline.fit(train_temp_df[SELECTED_FEATURES], train_temp_df[TARGET_COL].values)
        temporal_preds = pipeline.predict(test_temp_df[SELECTED_FEATURES])

        temporal_r2 = float(r2_score(test_temp_df[TARGET_COL].values, temporal_preds))
        temporal_rmse = float(np.sqrt(mean_squared_error(test_temp_df[TARGET_COL].values, temporal_preds)))
        temporal_mae = float(mean_absolute_error(test_temp_df[TARGET_COL].values, temporal_preds))

        # 4. Log Metrics to MLflow
        mlflow.log_metric("group_cv_r2", mean_gkf_r2)
        mlflow.log_metric("group_cv_rmse", mean_gkf_rmse)
        mlflow.log_metric("temporal_r2", temporal_r2)
        mlflow.log_metric("temporal_rmse", temporal_rmse)
        mlflow.log_metric("temporal_mae", temporal_mae)

        print(f"   -> Metrics: GroupKFold R2 = {mean_gkf_r2:.3f} | Temporal R2 = {temporal_r2:.3f}")

        # 5. Fit on Full Dataset & Log Model Artifact
        pipeline.fit(X, y)

        input_sample = X.iloc[:5]
        signature = infer_signature(input_sample, pipeline.predict(input_sample))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_sample.iloc[:1],
            serialization_format="cloudpickle",
        )
        print("   -> Logged model artifact with signature to MLflow")

        # 6. Save locally for FastAPI and Streamlit
        joblib.dump(pipeline, MODEL_ARTIFACT_PATH, compress=3)
        print(f"   -> Saved production pipeline artifact to {MODEL_ARTIFACT_PATH}")

    print(f"[SUCCESS] Finished run '{run_name}'.")


def train_ridge(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 1: Ridge Regression (Linear Baseline)."""
    alpha = 1.0
    params = {
        "model_type": "Ridge Regression",
        "alpha": alpha,
        "scaler": "StandardScaler",
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha, random_state=42)),
    ])
    evaluate_and_log(pipeline, "01_Ridge_Baseline", params, df, X, y)


def train_random_forest(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 2: Random Forest Regressor (Tree Ensemble)."""
    rf_params = {
        "model_type": "Random Forest Regressor",
        "n_estimators": 250,
        "max_depth": 18,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
    }
    # Random Forest splits orthogonally and does not require feature scaling
    pipeline = Pipeline([
        ("regressor", RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params["max_features"],
            random_state=rf_params["random_state"],
            n_jobs=-1,
        ))
    ])
    evaluate_and_log(pipeline, "02_Random_Forest", rf_params, df, X, y)


def train_lightgbm(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 3: LightGBM Regressor (Gradient Boosting).

    WHY LightGBM?
    - Boosting builds trees sequentially — each one corrects errors of the previous.
    - Leaf-wise growth (not level-wise) = deeper splits where it matters most.
    - Native L1/L2 regularization + row/col subsampling = built-in overfitting control.
    - Faster than XGBoost on most tabular datasets due to histogram-based splits.

    KEY HYPERPARAMETERS EXPLAINED:
    - n_estimators=300:  More rounds than RF (250) because each tree is weak (shallow).
                          Boosting needs more iterations to converge.
    - max_depth=6:       Shallow trees! Unlike RF (depth=18), boosting wants weak learners.
                          Deep trees in boosting = overfitting on small datasets (1686 rows).
    - learning_rate=0.05: How much each tree's prediction contributes.
                          Lower = more stable but needs more trees. 0.05 is conservative.
    - subsample=0.8:     Row sampling per tree. 80% adds stochastic regularization.
    - colsample_bytree=0.8: Column sampling. Prevents over-reliance on dominant features
                          like en_per_cap (energy use).
    - reg_alpha=0.1:     L1 (Lasso) penalty. Pushes weak feature contributions to zero.
    - reg_lambda=1.0:    L2 (Ridge) penalty. Shrinks all leaf weights — like having Ridge
                          regularization inside the boosting loop.
    - min_child_samples=5: Minimum rows in a leaf. With only 1686 rows and 95 countries,
                          setting this to 5 prevents leaf nodes from memorizing outliers.
    """
    lgbm_params = {
        "model_type": "LightGBM Regressor",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 5,
        "random_state": 42,
    }
    # LightGBM is tree-based — no scaling needed (same as Random Forest)
    pipeline = Pipeline([
        ("regressor", LGBMRegressor(
            n_estimators=lgbm_params["n_estimators"],
            max_depth=lgbm_params["max_depth"],
            learning_rate=lgbm_params["learning_rate"],
            subsample=lgbm_params["subsample"],
            colsample_bytree=lgbm_params["colsample_bytree"],
            reg_alpha=lgbm_params["reg_alpha"],
            reg_lambda=lgbm_params["reg_lambda"],
            min_child_samples=lgbm_params["min_child_samples"],
            random_state=lgbm_params["random_state"],
            verbosity=-1,  # Suppress LightGBM training logs
            n_jobs=-1,
        ))
    ])
    evaluate_and_log(pipeline, "03_LightGBM", lgbm_params, df, X, y)


def train_xgboost(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 4: XGBoost Regressor (eXtreme Gradient Boosting).

    WHY XGBoost?
    - Uses exact/histogram-based level-wise tree growth (breadth-first),
      which tends to produce more symmetric, balanced trees compared to LightGBM.
    - Employs 2nd-order Taylor expansions (both Gradient and Hessian) for objective
      optimization, making loss minimization mathematically very precise.
    - Robust against overfitting on medium-sized tabular datasets when tuned with
      subsampling and depth limits.

    KEY HYPERPARAMETERS EXPLAINED:
    - n_estimators=250:   Number of boosting rounds.
    - max_depth=6:        Maximum depth of a tree. Level-wise expansion means depth 6
                          creates up to 2^6 = 64 leaves in a balanced layout.
    - learning_rate=0.05: Shrinkage step size (eta). Prevents individual trees
                          from dominating predictions.
    - subsample=0.8:      Fraction of country-year observations sampled per round.
    - colsample_bytree=0.8: Fraction of features considered per tree split.
                          Prevents over-indexing on `en_per_cap`.
    - reg_alpha=0.1:      L1 regularization on leaf weights (sparsity).
    - reg_lambda=1.0:     L2 regularization on leaf weights (smoothness).
    - random_state=42:    Reproducibility.
    """
    xgb_params = {
        "model_type": "XGBoost Regressor",
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    pipeline = Pipeline([
        ("regressor", XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            reg_alpha=xgb_params["reg_alpha"],
            reg_lambda=xgb_params["reg_lambda"],
            random_state=xgb_params["random_state"],
            tree_method="hist",
            n_jobs=-1,
        ))
    ])
    evaluate_and_log(pipeline, "04_XGBoost", xgb_params, df, X, y)


def train_voting_ensemble(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 5: Voting Ensemble (Weighted Blending of Diverse Architectures).

    WHY ENSEMBLE LEARNING?
    - Individual algorithms have distinct inductive biases:
      * Ridge: Global smooth hyperplanes (low variance, high bias on non-linearities)
      * Random Forest: Bagged axis-aligned orthogonal partitions (low variance, robust to noise)
      * LightGBM: Sequential gradient-guided leaf-wise splits (low bias, captures non-linear interactions)
    - By taking a weighted average of these distinct architectures, uncorrelated prediction errors
      cancel each other out (Condorcet's Jury Theorem / Jensen's Inequality).
    - Weight allocation [1, 2, 2]:
      * Ridge (1x): Acts as a smooth regularizer that anchors boundary predictions
      * Random Forest (2x): Stable, non-parametric interpolator
      * LightGBM (2x): Highly adaptive non-linear predictor
    """
    params = {
        "model_type": "Voting Ensemble (Ridge + RF + LightGBM)",
        "weights": [1, 2, 2],
        "base_estimators": "Ridge(StandardScaler) + RandomForest + LightGBM",
    }

    # 1. Base Sub-Pipelines
    ridge_sub = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    rf_sub = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    lgb_sub = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # 2. Combine into VotingRegressor
    ensemble = VotingRegressor(
        estimators=[
            ("ridge", ridge_sub),
            ("rf", rf_sub),
            ("lgb", lgb_sub),
        ],
        weights=params["weights"],
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("regressor", ensemble)
    ])

    evaluate_and_log(pipeline, "05_Voting_Ensemble", params, df, X, y)


def train_stacking_ensemble(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray):
    """MODEL 6: Stacking Ensemble (Meta-Learning on Out-of-Fold Predictions).

    WHY STACKING?
    - Voting uses fixed human-picked weights (e.g., [1, 2, 2]).
    - Stacking replaces manual weights with an ML model (Meta-Estimator / Level-1):
      * Level 0: Diverse base learners (Ridge, Random Forest, LightGBM) fit the raw features.
      * Cross-Validation (cv=3): Each base learner generates out-of-fold validation predictions.
      * Level 1: A meta-regressor (Ridge with L2 regularization) is trained on those out-of-fold
        predictions:  y_pred = b0 + b1*(y_ridge) + b2*(y_rf) + b3*(y_lgb)
    - Crucial Leakage Prevention: Level-1 MUST be trained on *out-of-fold* predictions,
      not in-sample predictions, otherwise the meta-learner severely overfits to the base
      learners with lower training error.
    """
    params = {
        "model_type": "Stacking Ensemble (Meta: Ridge)",
        "meta_estimator": "Ridge(alpha=1.0)",
        "base_estimators": "Ridge(StandardScaler) + RandomForest + LightGBM",
        "internal_cv": 3,
    }

    # 1. Base Estimators (Level 0)
    ridge_sub = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    rf_sub = RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    lgb_sub = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    # 2. Meta Estimator (Level 1)
    meta_regressor = Ridge(alpha=1.0, random_state=42)

    ensemble = StackingRegressor(
        estimators=[
            ("ridge", ridge_sub),
            ("rf", rf_sub),
            ("lgb", lgb_sub),
        ],
        final_estimator=meta_regressor,
        cv=params["internal_cv"],
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("regressor", ensemble)
    ])

    evaluate_and_log(pipeline, "06_Stacking_Ensemble", params, df, X, y)


def main():
    parser = argparse.ArgumentParser(description="Train and track models in MLflow.")
    parser.add_argument(
        "--model",
        choices=["ridge", "rf", "lgbm", "xgb", "voting", "stacking", "all"],
        default="stacking",
        help="Model to train: 'ridge', 'rf', 'lgbm', 'xgb', 'voting', 'stacking', or 'all'",
    )
    args = parser.parse_args()

    # Set active MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"[EXPERIMENT] Active MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")

    # Load data
    df = load_dataset()
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COL].values
    print(f"[DATA] Loaded {len(df)} rows across {df[COUNTRY_COL].nunique()} countries.")

    if args.model == "ridge":
        train_ridge(df, X, y)
    elif args.model == "rf":
        train_random_forest(df, X, y)
    elif args.model == "lgbm":
        train_lightgbm(df, X, y)
    elif args.model == "xgb":
        train_xgboost(df, X, y)
    elif args.model == "voting":
        train_voting_ensemble(df, X, y)
    elif args.model == "stacking":
        train_stacking_ensemble(df, X, y)
    elif args.model == "all":
        train_ridge(df, X, y)
        train_random_forest(df, X, y)
        train_lightgbm(df, X, y)
        train_xgboost(df, X, y)
        train_voting_ensemble(df, X, y)
        train_stacking_ensemble(df, X, y)


if __name__ == "__main__":
    main()
