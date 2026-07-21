"""Prediction utilities for upcoming fixtures."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from penaltyblog.models.match_outcome_model import OUTCOME_LABELS, MatchOutcomeModel


def generate_fixture_predictions(
    model: MatchOutcomeModel,
    fixtures_df: pd.DataFrame,
    feature_columns: list[str],
    home_odds_col: Optional[str] = None,
    draw_odds_col: Optional[str] = None,
    away_odds_col: Optional[str] = None,
) -> pd.DataFrame:
    """Generate fixture-level probabilities and confidence scores."""
    X = fixtures_df[feature_columns].to_numpy(dtype=float)
    probs = model.predict_proba(X)

    out = fixtures_df.copy()
    out["prob_home_win"] = probs[:, 0]
    out["prob_draw"] = probs[:, 1]
    out["prob_away_win"] = probs[:, 2]
    out["predicted_outcome"] = OUTCOME_LABELS[np.argmax(probs, axis=1)]
    out["confidence_score"] = probs.max(axis=1)

    if home_odds_col and draw_odds_col and away_odds_col:
        implied = np.column_stack(
            [
                1.0 / out[home_odds_col].to_numpy(dtype=float),
                1.0 / out[draw_odds_col].to_numpy(dtype=float),
                1.0 / out[away_odds_col].to_numpy(dtype=float),
            ]
        )
        implied = implied / implied.sum(axis=1, keepdims=True)
        out["market_edge_home"] = out["prob_home_win"] - implied[:, 0]
        out["market_edge_draw"] = out["prob_draw"] - implied[:, 1]
        out["market_edge_away"] = out["prob_away_win"] - implied[:, 2]

    return out
