import numpy as np
import pandas as pd

from penaltyblog.pipelines import (
    engineer_match_features,
    evaluate_predictions,
    generate_fixture_predictions,
    temporal_train_test_split,
    train_match_outcome_pipeline,
)


def _matches_df():
    return pd.DataFrame(
        {
            "match_id": list(range(1, 13)),
            "match_date": pd.date_range("2023-01-01", periods=12, freq="7D"),
            "home_team": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "away_team": ["B", "C", "A", "C", "A", "B", "B", "C", "A", "C", "A", "B"],
            "home_goals": [2, 1, 0, 3, 1, 2, 1, 0, 1, 2, 2, 1],
            "away_goals": [1, 1, 2, 0, 1, 1, 0, 2, 1, 1, 0, 2],
            "home_shots": [10, 8, 6, 11, 7, 9, 8, 5, 7, 10, 9, 6],
            "away_shots": [7, 8, 9, 5, 7, 8, 6, 10, 7, 8, 6, 9],
            "home_xg": [1.8, 1.1, 0.8, 2.2, 1.0, 1.6, 1.3, 0.7, 1.0, 1.9, 1.7, 0.9],
            "away_xg": [1.0, 1.0, 1.6, 0.6, 1.1, 1.0, 0.7, 1.9, 1.1, 1.1, 0.8, 1.8],
            "home_possession": [55, 51, 48, 57, 50, 53, 52, 47, 50, 56, 54, 49],
            "away_possession": [45, 49, 52, 43, 50, 47, 48, 53, 50, 44, 46, 51],
        }
    )


def test_feature_engineering_training_prediction_evaluation_flow():
    feats = engineer_match_features(_matches_df())
    raw = _matches_df()

    for team in ["A", "B", "C"]:
        home_codes = feats.loc[raw["home_team"] == team, "home_team_code"]
        away_codes = feats.loc[raw["away_team"] == team, "away_team_code"]
        assert set(home_codes.unique()) == set(away_codes.unique())

    feature_columns = [
        "home_shots",
        "away_shots",
        "home_xg",
        "away_xg",
        "home_possession",
        "away_possession",
        "goal_differential",
        "home_shot_efficiency",
        "away_shot_efficiency",
        "home_defensive_strength",
        "away_defensive_strength",
        "home_form",
        "away_form",
        "home_team_code",
        "away_team_code",
    ]

    train_df, test_df = temporal_train_test_split(feats, test_size=0.25)
    assert train_df["match_date"].max() <= test_df["match_date"].min()

    bundle = train_match_outcome_pipeline(feats, feature_columns)
    preds = generate_fixture_predictions(bundle.model, feats, feature_columns)

    assert {"prob_home_win", "prob_draw", "prob_away_win", "confidence_score"}.issubset(
        preds.columns
    )
    probs = preds[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)

    metrics = evaluate_predictions(probs, feats["outcome"].to_numpy())
    assert "rps" in metrics
    assert "ece" in metrics
    assert metrics["confusion_matrix"].shape == (3, 3)
