"""Training utilities for end-to-end match outcome prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from penaltyblog.models.match_outcome_model import MatchOutcomeModel


@dataclass
class ModelBundle:
    """Persistable trained model bundle."""

    model: MatchOutcomeModel
    feature_columns: list[str]
    metadata: dict[str, Any]


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "match_date",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test using chronological order."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * (1 - test_size)))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def train_match_outcome_pipeline(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    outcome_col: str = "outcome",
    bayesian: bool = False,
    random_state: Optional[int] = 42,
) -> ModelBundle:
    """Train model with temporal validation and simple hyperparameter search."""
    train_df, valid_df = temporal_train_test_split(features_df)

    if len(valid_df) == 0:
        raise ValueError("Validation split is empty. Add more matches.")

    best_model: Optional[MatchOutcomeModel] = None
    best_loss = float("inf")
    best_reg = None

    regs = [0.01, 0.1, 1.0]
    for reg in regs:
        model = (
            MatchOutcomeModel.bayesian_variant(
                regularization=reg,
                prior_variance=5.0,
                random_state=random_state,
            )
            if bayesian
            else MatchOutcomeModel.frequentist(
                regularization=reg,
                random_state=random_state,
            )
        )
        model.fit(
            train_df[feature_columns].to_numpy(dtype=float),
            train_df[outcome_col].to_numpy(dtype=int),
            feature_names=feature_columns,
        )
        probs = model.predict_proba(valid_df[feature_columns].to_numpy(dtype=float))
        y = valid_df[outcome_col].to_numpy(dtype=int)
        loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_reg = reg

    assert best_model is not None

    best_model.calibrate_temperature(
        valid_df[feature_columns].to_numpy(dtype=float),
        valid_df[outcome_col].to_numpy(dtype=int),
    )

    metadata = {
        "best_regularization": best_reg,
        "validation_log_loss": best_loss,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "variant": "bayesian" if bayesian else "frequentist",
        "random_state": random_state,
    }
    return ModelBundle(model=best_model, feature_columns=feature_columns, metadata=metadata)


def save_model_bundle(bundle: ModelBundle, path: str) -> None:
    """Save a model bundle with metadata."""
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: str) -> ModelBundle:
    """Load a model bundle with metadata."""
    with open(path, "rb") as f:
        return pickle.load(f)
