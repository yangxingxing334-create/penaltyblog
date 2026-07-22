"""Pipelines for data, training, prediction, and evaluation."""

from .data_sources import (
    FEATURE_CONTRACT_COLUMNS,
    OPENFOOTBALL_BRAZIL_SERIE_A_URL_TEMPLATE,
    ODDS_COLUMNS,
    REQUIRED_CONTRACT_COLUMNS,
    DataSourceResult,
    QualityGateReport,
    data_source_metrics,
    ensure_contract_columns,
    fetch_opta_contract_matches,
    fetch_statsbomb_contract_matches,
    fetch_with_fallback,
    load_snapshot,
    normalize_source_frame,
    run_quality_gates,
    save_daily_snapshot,
    split_historical_and_target_fixtures,
    fetch_external_brazil_serie_a_contract_matches,
    validate_data_contract,
)
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
    "DataSourceResult",
    "FEATURE_CONTRACT_COLUMNS",
    "ModelBundle",
    "ODDS_COLUMNS",
    "OPENFOOTBALL_BRAZIL_SERIE_A_URL_TEMPLATE",
    "QualityGateReport",
    "REQUIRED_CONTRACT_COLUMNS",
    "data_source_metrics",
    "engineer_match_features",
    "ensure_contract_columns",
    "evaluate_predictions",
    "fetch_opta_contract_matches",
    "fetch_statsbomb_matches",
    "fetch_statsbomb_contract_matches",
    "fetch_with_fallback",
    "generate_fixture_predictions",
    "load_model_bundle",
    "load_snapshot",
    "market_profitability_analysis",
    "normalize_source_frame",
    "run_quality_gates",
    "save_daily_snapshot",
    "fetch_external_brazil_serie_a_contract_matches",
    "split_historical_and_target_fixtures",
    "save_model_bundle",
    "temporal_train_test_split",
    "train_match_outcome_pipeline",
    "validate_data_contract",
]
