"""Feature engineering for football 1X2 classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace(0, np.nan)
    return (num / den2).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _encode_outcome(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    return np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))


def engineer_match_features(
    matches: pd.DataFrame,
    rolling_window: int = 5,
    normalize: bool = True,
) -> pd.DataFrame:
    """Engineer match features for 1X2 prediction.

    Expects at least: match_date, home_team, away_team, home_goals, away_goals.
    Optional columns: home_shots, away_shots, home_xg, away_xg, home_possession,
    away_possession.
    """
    required = ["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    for col in [
        "home_shots",
        "away_shots",
        "home_xg",
        "away_xg",
        "home_possession",
        "away_possession",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    df["goal_differential"] = df["home_goals"] - df["away_goals"]
    df["home_shot_efficiency"] = _safe_ratio(df["home_goals"], df["home_shots"])
    df["away_shot_efficiency"] = _safe_ratio(df["away_goals"], df["away_shots"])
    df["home_defensive_strength"] = _safe_ratio(1.0, 1.0 + df["away_xg"])
    df["away_defensive_strength"] = _safe_ratio(1.0, 1.0 + df["home_xg"])

    home_points = np.where(
        df["home_goals"] > df["away_goals"], 3, np.where(df["home_goals"] == df["away_goals"], 1, 0)
    )
    away_points = np.where(
        df["away_goals"] > df["home_goals"], 3, np.where(df["home_goals"] == df["away_goals"], 1, 0)
    )

    df["home_form"] = (
        pd.Series(home_points)
        .groupby(df["home_team"])
        .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
        .fillna(0.0)
    )
    df["away_form"] = (
        pd.Series(away_points)
        .groupby(df["away_team"])
        .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean())
        .fillna(0.0)
    )

    teams = pd.concat([df["home_team"], df["away_team"]], ignore_index=True).unique()
    team_categories = pd.Index(teams)
    df["home_team_code"] = pd.Categorical(df["home_team"], categories=team_categories).codes
    df["away_team_code"] = pd.Categorical(df["away_team"], categories=team_categories).codes
    df["outcome"] = _encode_outcome(df["home_goals"], df["away_goals"])

    numeric_cols: list[str] = [
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
    if normalize:
        for col in numeric_cols:
            std = df[col].std(ddof=0)
            if std > 0:
                df[col] = (df[col] - df[col].mean()) / std
            else:
                df[col] = 0.0

    return df
