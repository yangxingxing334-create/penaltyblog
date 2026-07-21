"""StatsBomb data acquisition utilities built on Flow.statsbomb."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from penaltyblog.matchflow import Flow


REQUIRED_COLUMNS = [
    "match_id",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]


def _normalise_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "match_id": "match_id",
        "match_date": "match_date",
        "home_team.home_team_name": "home_team",
        "away_team.away_team_name": "away_team",
        "home_score": "home_goals",
        "away_score": "away_goals",
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    return df


def fetch_statsbomb_matches(
    competition_id: int,
    season_id: int,
    creds: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_team_season_stats: bool = False,
) -> pd.DataFrame:
    """Fetch match-level data from StatsBomb through Flow.

    Returns a normalized DataFrame containing key match fields and optionally
    filtered by date range.
    """
    flow = Flow.statsbomb.matches(
        competition_id=competition_id,
        season_id=season_id,
        creds=creds,
    ).flatten()
    matches = _normalise_match_columns(flow.to_pandas())

    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"Missing required StatsBomb fields: {missing}")

    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "home_team", "away_team"])

    if start_date is not None:
        matches = matches[matches["match_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        matches = matches[matches["match_date"] <= pd.to_datetime(end_date)]

    matches = matches.reset_index(drop=True)

    if include_team_season_stats:
        stats = (
            Flow.statsbomb.team_season_stats(
                competition_id=competition_id,
                season_id=season_id,
                creds=creds,
            )
            .flatten()
            .to_pandas()
        )
        if not stats.empty:
            stats = stats.rename(
                columns={
                    "team.team_name": "team",
                    "team_name": "team",
                }
            )
            for col in ["team", "shots", "xg", "possession"]:
                if col not in stats.columns:
                    stats[col] = np.nan
            team_stats = stats[["team", "shots", "xg", "possession"]].copy()
            home_stats = team_stats.rename(
                columns={
                    "team": "home_team",
                    "shots": "home_shots",
                    "xg": "home_xg",
                    "possession": "home_possession",
                }
            )
            away_stats = team_stats.rename(
                columns={
                    "team": "away_team",
                    "shots": "away_shots",
                    "xg": "away_xg",
                    "possession": "away_possession",
                }
            )
            matches = matches.merge(home_stats, on="home_team", how="left")
            matches = matches.merge(away_stats, on="away_team", how="left")

    return matches
