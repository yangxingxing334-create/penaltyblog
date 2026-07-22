"""Production data-source utilities for reproducible prediction inputs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from penaltyblog.matchflow import Flow

REQUIRED_CONTRACT_COLUMNS = [
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]
FEATURE_CONTRACT_COLUMNS = [
    "home_shots",
    "away_shots",
    "home_xg",
    "away_xg",
    "home_possession",
    "away_possession",
]
ODDS_COLUMNS = ["home_odds", "draw_odds", "away_odds"]

_COLUMN_ALIASES = {
    "match_id": "match_id",
    "fixture_uuid": "match_id",
    "fixture.id": "match_id",
    "fixture_uuid_id": "match_id",
    "match_date": "match_date",
    "date": "match_date",
    "utc_date": "match_date",
    "home_team.home_team_name": "home_team",
    "away_team.away_team_name": "away_team",
    "home.team.name": "home_team",
    "away.team.name": "away_team",
    "home_score": "home_goals",
    "away_score": "away_goals",
    "home_goals": "home_goals",
    "away_goals": "away_goals",
    "home_shots": "home_shots",
    "away_shots": "away_shots",
    "home_xg": "home_xg",
    "away_xg": "away_xg",
    "home_possession": "home_possession",
    "away_possession": "away_possession",
    "home_odds": "home_odds",
    "draw_odds": "draw_odds",
    "away_odds": "away_odds",
}


@dataclass
class QualityGateReport:
    """Result for completeness/consistency/sanity checks."""

    completeness_ok: bool
    consistency_ok: bool
    sanity_ok: bool
    issues: list[str]

    @property
    def passed(self) -> bool:
        return self.completeness_ok and self.consistency_ok and self.sanity_ok


@dataclass
class DataSourceResult:
    """Result payload for primary/fallback acquisition."""

    source: str
    dataframe: pd.DataFrame
    latency_seconds: float
    used_fallback: bool
    degraded_mode: bool
    quality_report: QualityGateReport


def normalize_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source-specific columns into the unified data contract."""
    out = df.copy()
    rename_map = {c: _COLUMN_ALIASES[c] for c in out.columns if c in _COLUMN_ALIASES}
    out = out.rename(columns=rename_map)
    if "match_date" in out.columns:
        out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    return out


def ensure_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required + optional contract columns exist."""
    out = df.copy()
    for col in FEATURE_CONTRACT_COLUMNS + ODDS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def validate_data_contract(df: pd.DataFrame, require_odds: bool = False) -> None:
    """Validate mandatory schema for downstream feature engineering."""
    missing = [c for c in REQUIRED_CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required contract columns: {missing}")

    if require_odds:
        missing_odds = [c for c in ODDS_COLUMNS if c not in df.columns]
        if missing_odds:
            raise ValueError(f"Missing required odds columns: {missing_odds}")

    if df["match_date"].isna().any():
        raise ValueError("match_date contains null/invalid values")
    if df["home_team"].isna().any() or df["away_team"].isna().any():
        raise ValueError("home_team/away_team contains null values")


def run_quality_gates(
    df: pd.DataFrame,
    expected_match_count: Optional[int] = None,
) -> QualityGateReport:
    """Run completeness/consistency/sanity checks before snapshot/prediction."""
    issues: list[str] = []

    completeness_ok = True
    if expected_match_count is not None and len(df) != expected_match_count:
        completeness_ok = False
        issues.append(
            f"Expected {expected_match_count} fixtures but received {len(df)}"
        )

    consistency_ok = True
    if (df["home_team"] == df["away_team"]).any():
        consistency_ok = False
        issues.append("Detected fixtures where home_team == away_team")
    if "match_id" in df.columns and df["match_id"].notna().any():
        dupes = int(df["match_id"].duplicated().sum())
        if dupes > 0:
            consistency_ok = False
            issues.append(f"Detected {dupes} duplicate match_id values")

    sanity_ok = True
    for goals_col in ("home_goals", "away_goals"):
        if (pd.to_numeric(df[goals_col], errors="coerce") < 0).any():
            sanity_ok = False
            issues.append(f"Negative values found in {goals_col}")
    for odds_col in ODDS_COLUMNS:
        if odds_col in df.columns and df[odds_col].notna().any():
            bad = (pd.to_numeric(df[odds_col], errors="coerce") <= 1.0).sum()
            if bad:
                sanity_ok = False
                issues.append(f"Found {int(bad)} invalid {odds_col} values (<= 1.0)")

    return QualityGateReport(
        completeness_ok=completeness_ok,
        consistency_ok=consistency_ok,
        sanity_ok=sanity_ok,
        issues=issues,
    )


def fetch_statsbomb_contract_matches(
    competition_id: int,
    season_id: int,
    creds: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_team_season_stats: bool = True,
) -> pd.DataFrame:
    """Fetch StatsBomb matches and map them into the contract schema."""
    from .statsbomb_fetcher import fetch_statsbomb_matches

    matches = fetch_statsbomb_matches(
        competition_id=competition_id,
        season_id=season_id,
        creds=creds,
        start_date=start_date,
        end_date=end_date,
        include_team_season_stats=include_team_season_stats,
    )
    return ensure_contract_columns(normalize_source_frame(matches))


def fetch_opta_contract_matches(
    tournament_calendar_uuid: str,
    creds: Optional[dict] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    use_opta_names: bool = True,
) -> pd.DataFrame:
    """Fetch Opta matches and map them into the contract schema."""
    flow = Flow.opta.matches(
        tournament_calendar_uuid=tournament_calendar_uuid,
        creds=creds,
        date_from=date_from,
        date_to=date_to,
        use_opta_names=use_opta_names,
        live=False,
    ).flatten()
    df = flow.to_pandas()
    out = ensure_contract_columns(normalize_source_frame(df))
    if out["home_goals"].isna().any():
        out["home_goals"] = out["home_goals"].fillna(0)
    if out["away_goals"].isna().any():
        out["away_goals"] = out["away_goals"].fillna(0)
    return out


def fetch_with_fallback(
    primary_fetcher: Callable[[], pd.DataFrame],
    fallback_fetcher: Optional[Callable[[], pd.DataFrame]] = None,
    *,
    max_latency_seconds: float = 20.0,
    expected_match_count: Optional[int] = None,
    require_odds_for_betting: bool = False,
) -> DataSourceResult:
    """Fetch from primary first, then fail over on error/latency/quality issues."""
    errors: list[str] = []

    for idx, (name, fetcher) in enumerate(
        [("primary", primary_fetcher), ("fallback", fallback_fetcher)]
    ):
        if fetcher is None:
            continue

        t0 = time.perf_counter()
        try:
            frame = normalize_source_frame(fetcher())
            frame = ensure_contract_columns(frame)
            latency = time.perf_counter() - t0
            if latency > max_latency_seconds:
                raise TimeoutError(
                    f"{name} latency {latency:.2f}s exceeded {max_latency_seconds:.2f}s"
                )
            validate_data_contract(frame, require_odds=require_odds_for_betting)
            report = run_quality_gates(frame, expected_match_count=expected_match_count)
            if not report.passed:
                raise ValueError("; ".join(report.issues))

            degraded_mode = require_odds_for_betting and frame[ODDS_COLUMNS].isna().any(
                axis=None
            )
            return DataSourceResult(
                source=name,
                dataframe=frame,
                latency_seconds=latency,
                used_fallback=idx == 1,
                degraded_mode=degraded_mode,
                quality_report=report,
            )
        except (TimeoutError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            errors.append(f"{name} failed: {exc}")

    raise RuntimeError("No data source available. " + " | ".join(errors))


def save_daily_snapshot(
    df: pd.DataFrame,
    snapshot_dir: str,
    *,
    dataset_name: str = "kleague",
    as_of_date: Optional[str] = None,
) -> dict[str, str]:
    """Persist a reproducible daily input snapshot and metadata."""
    ts = datetime.now(timezone.utc)
    date_tag = as_of_date or ts.strftime("%Y-%m-%d")
    base = Path(snapshot_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    csv_path = base / f"{dataset_name}_{date_tag}.csv"
    metadata_path = base / f"{dataset_name}_{date_tag}.metadata.json"

    df.to_csv(csv_path, index=False)
    metadata = {
        "dataset_name": dataset_name,
        "as_of_date": date_tag,
        "created_at_utc": ts.isoformat(),
        "row_count": int(len(df)),
        "columns": list(df.columns),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return {"csv": str(csv_path), "metadata": str(metadata_path)}


def load_snapshot(csv_path: str) -> pd.DataFrame:
    """Load a saved snapshot for reproducible prediction runs."""
    df = pd.read_csv(csv_path)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def data_source_metrics(result: DataSourceResult) -> dict[str, object]:
    """Build metrics payload for logging/alerting integrations."""
    return {
        "source": result.source,
        "used_fallback": result.used_fallback,
        "degraded_mode": result.degraded_mode,
        "latency_seconds": result.latency_seconds,
        "quality_report": asdict(result.quality_report),
    }
