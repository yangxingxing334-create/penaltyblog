"""Pipelines for data, training, prediction, and evaluation."""

from .evaluator import evaluate_predictions, market_profitability_analysis
from .feature_engineer import engineer_match_features
from .predictor import generate_fixture_predictions
from .statsbomb_fetcher import fetch_statsbomb_matches
from .training import (
    ModelBundle,
    load_model_bundle,
    save_model_bundle,
    temporal_train_test_split,
    train_match_outcome_pipeline,
)

__all__ = [
    "ModelBundle",
    "engineer_match_features",
    "evaluate_predictions",
    "fetch_statsbomb_matches",
    "generate_fixture_predictions",
    "load_model_bundle",
    "market_profitability_analysis",
    "save_model_bundle",
    "temporal_train_test_split",
    "train_match_outcome_pipeline",
]
