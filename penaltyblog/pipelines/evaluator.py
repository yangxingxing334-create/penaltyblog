"""Evaluation metrics for 1X2 football predictions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from penaltyblog.metrics import rps_average


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3
) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def _expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        is_last_bin = i == len(bins) - 2
        if is_last_bin:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = np.mean(predictions[mask] == y_true[mask])
        conf = np.mean(confidences[mask])
        ece += np.abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def market_profitability_analysis(
    probs: np.ndarray,
    y_true: np.ndarray,
    decimal_odds: np.ndarray,
) -> dict[str, float]:
    """Estimate expected and realized ROI using max-edge selection."""
    implied = 1.0 / decimal_odds
    implied = implied / implied.sum(axis=1, keepdims=True)
    edges = probs - implied
    picks = edges.argmax(axis=1)

    pick_odds = decimal_odds[np.arange(len(picks)), picks]
    pick_probs = probs[np.arange(len(picks)), picks]
    expected_return = np.mean(pick_odds * pick_probs - 1.0)
    won = (picks == y_true).astype(float)
    realized_return = np.mean(won * pick_odds - 1.0)
    return {
        "expected_roi": float(expected_return),
        "realized_roi": float(realized_return),
    }


def evaluate_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    decimal_odds: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Compute RPS, calibration, confusion matrix, and optional profitability."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)

    out: dict[str, object] = {
        "rps": float(rps_average(probs, y_true)),
        "brier": float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1))),
        "ece": _expected_calibration_error(probs, y_true),
        "confusion_matrix": pd.DataFrame(
            _confusion_matrix(y_true, pred),
            index=["home_win", "draw", "away_win"],
            columns=["home_win", "draw", "away_win"],
        ),
    }

    if decimal_odds is not None:
        out["market_profitability"] = market_profitability_analysis(
            probs,
            y_true,
            np.asarray(decimal_odds, dtype=float),
        )

    return out
