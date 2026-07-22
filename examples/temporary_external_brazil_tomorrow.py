"""Temporary external-mode prediction for tomorrow's Brazil Serie A fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from penaltyblog.pipelines import (
    fetch_external_brazil_serie_a_contract_matches,
    generate_fixture_predictions,
    engineer_match_features,
    split_historical_and_target_fixtures,
    train_match_outcome_pipeline,
)

FEATURE_COLUMNS = [
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


def _tomorrow_utc() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()


def run_temporary_external_brazil_tomorrow_prediction(
    *,
    target_date: Optional[str] = None,
    season: Optional[int] = None,
    source_url: Optional[str] = None,
    team_mappings: Optional[dict] = None,
) -> dict[str, object]:
    """Run temporary external-mode Brazil Serie A predictions for a target date."""
    requested_date = target_date or _tomorrow_utc()
    target_season = season or int(requested_date[:4])

    try:
        data = fetch_external_brazil_serie_a_contract_matches(
            season=target_season,
            source_url=source_url,
            team_mappings=team_mappings,
        )
        historical, fixtures = split_historical_and_target_fixtures(data, requested_date)
        if fixtures.empty:
            raise RuntimeError(
                f"No Brazil Serie A fixtures found for {requested_date} in temporary external source"
            )
        if historical.empty:
            raise RuntimeError("No historical completed matches available for training")

        training_features = engineer_match_features(historical, normalize=False)
        bundle = train_match_outcome_pipeline(training_features, FEATURE_COLUMNS, random_state=42)

        merged = pd.concat(
            [historical.assign(_is_fixture=False), fixtures.assign(_is_fixture=True)],
            ignore_index=True,
        ).sort_values("match_date")
        merged_features = engineer_match_features(
            merged.drop(columns=["_is_fixture"]),
            normalize=False,
        )
        fixture_features = merged_features[merged["_is_fixture"].to_numpy()]
        preds = generate_fixture_predictions(bundle.model, fixture_features, FEATURE_COLUMNS)

        result = preds[
            [
                "match_date",
                "home_team",
                "away_team",
                "prob_home_win",
                "prob_draw",
                "prob_away_win",
                "predicted_outcome",
            ]
        ].copy()
        result["match_date"] = pd.to_datetime(result["match_date"]).dt.date.astype(str)
        return {
            "status": "success",
            "mode": "temporary_external",
            "source": "openfootball",
            "target_date": requested_date,
            "season": target_season,
            "predictions": result.to_dict(orient="records"),
            "model_metadata": bundle.metadata,
        }
    except Exception as exc:  # pragma: no cover - runtime integration path
        return {
            "status": "failed",
            "mode": "temporary_external",
            "source": "openfootball",
            "target_date": requested_date,
            "season": target_season,
            "error": str(exc),
            "retry_suggestions": [
                "确认 source_url 可访问，或稍后重试",
                "确认 season 与 target_date 对应赛季一致",
                "如仍失败，切换回仓库内正式数据源流程",
            ],
        }


if __name__ == "__main__":
    output = run_temporary_external_brazil_tomorrow_prediction()
    print(output)
