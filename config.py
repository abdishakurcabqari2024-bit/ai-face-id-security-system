"""
Config - AI Face ID Security System
Lead AI Machine Learning Architecture Configuration
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"
INTRUDER_DIR = LOGS_DIR / "intruders"

# Kaggle Dataset Path (Cleaned 31 Classes, 2,557 Images)
KAGGLE_DATASET_DIR = Path(r"C:\Users\abdiz\Downloads\archive\dataset_prepared")
CACHE_DIR = DATA_DIR / "cache"

for d in [DATA_DIR, FACES_DIR, MODELS_DIR, LOGS_DIR, INTRUDER_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Face Processing Dimensions
FACE_WIDTH = 128
FACE_HEIGHT = 128

# Security & Verification Thresholds
# Minimum probability score required to grant access
DEFAULT_SECURITY_THRESHOLD = 0.75  # 75% confidence minimum
INTRUDER_ALERT_THRESHOLD = 0.50

# Ensemble Weights for Decision Making
# XGBoost weight vs Random Forest weight
WEIGHT_XGBOOST = 0.55
WEIGHT_RANDOM_FOREST = 0.45

# Liveness Detection Config
LIVENESS_BLUR_THRESHOLD = 85.0  # Laplacian variance minimum (filters flat 2D prints)
REQUIRE_LIVENESS = True

# Feature Extraction Settings
HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_RADIUS = 2
LBP_POINTS = 16
PCA_COMPONENTS = 64  # Condensed high-signal dimensional representation

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 150,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'random_state': 42
}

# Random Forest Hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'criterion': 'gini',
    'n_jobs': -1,
    'random_state': 42
}

# Database & Logging
DB_PATH = LOGS_DIR / "security_audit.db"
CSV_LOG_PATH = LOGS_DIR / "access_log.csv"

# Camera Settings
CAMERA_INDEX = 0
FRAME_RATE = 30
