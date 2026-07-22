"""Complete end-to-end StatsBomb 1X2 prediction demo."""

from __future__ import annotations

from penaltyblog.pipelines import (
    engineer_match_features,
    evaluate_predictions,
    fetch_statsbomb_matches,
    generate_fixture_predictions,
    train_match_outcome_pipeline,
)


def run_demo(
    competition_id: int,
    season_id: int,
    creds: dict | None = None,
) -> dict[str, object]:
    """Run the full workflow: fetch -> feature engineer -> train -> predict."""
    matches = fetch_statsbomb_matches(
        competition_id=competition_id,
        season_id=season_id,
        creds=creds,
    )
    features = engineer_match_features(matches)

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

    bundle = train_match_outcome_pipeline(features, feature_columns)
    predictions = generate_fixture_predictions(bundle.model, features, feature_columns)

    metrics = evaluate_predictions(
        predictions[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy(),
        features["outcome"].to_numpy(),
    )
    return {
        "metadata": bundle.metadata,
        "predictions": predictions,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Example IDs (replace with desired competition and season).
    result = run_demo(competition_id=2, season_id=44)
    print(result["metadata"])
    print(result["metrics"])
