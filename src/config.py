from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data File Paths
PROCESSED_DATA_PATH = PROJECT_ROOT / "data_cleaned.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "co2_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

# Column Definitions
TARGET_COL = "co2_per_cap"
COUNTRY_COL = "country"
YEAR_COL = "year"

# All candidate economic, demographic, and agricultural features
FEATURE_COLS = [
    "cereal_yield",
    "fdi_perc_gdp",
    "gni_per_cap",
    "en_per_cap",
    "pop_urb_aggl_perc",
    "prot_area_perc",
    "pop_growth_perc",
    "urb_pop_growth_perc",
]

# Features selected after RFECV (excluding noisy fdi_perc_gdp)
SELECTED_FEATURES = [
    "cereal_yield",
    "gni_per_cap",
    "en_per_cap",
    "pop_urb_aggl_perc",
    "prot_area_perc",
    "pop_growth_perc",
    "urb_pop_growth_perc",
]

# Reproducibility & Splits
RANDOM_STATE = 42
TEMPORAL_SPLIT_YEAR = 2003  # Train <= 2003, Test > 2003
N_SPLITS_CV = 5
MLFLOW_EXPERIMENT_NAME = "carbon-emission-benchmark"

