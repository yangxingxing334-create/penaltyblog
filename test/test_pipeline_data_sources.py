import json

import pandas as pd
import pytest

from penaltyblog.pipelines.data_sources import (
    fetch_with_fallback,
    load_snapshot,
    run_quality_gates,
    save_daily_snapshot,
    validate_data_contract,
)


def _contract_df():
    return pd.DataFrame(
        {
            "match_id": [1, 2],
            "match_date": ["2026-07-22", "2026-07-22"],
            "home_team": ["FC Seoul", "Gwangju FC"],
            "away_team": ["Pohang Steelers", "Kimcheon Sangmu"],
            "home_goals": [0, 0],
            "away_goals": [0, 0],
        }
    )


def test_fetch_with_fallback_switches_on_primary_error():
    def _primary():
        raise RuntimeError("primary unavailable")

    def _fallback():
        return _contract_df()

    out = fetch_with_fallback(
        _primary,
        _fallback,
        expected_match_count=2,
        max_latency_seconds=5.0,
    )
    assert out.used_fallback is True
    assert out.source == "fallback"
    assert out.quality_report.passed is True


def test_fetch_with_fallback_switches_on_latency(monkeypatch):
    def _primary():
        return _contract_df()

    def _fallback():
        return _contract_df()

    ticks = iter([0.0, 1.0, 1.0, 1.1])
    monkeypatch.setattr(
        "penaltyblog.pipelines.data_sources.time.perf_counter",
        lambda: next(ticks),
    )

    out = fetch_with_fallback(
        _primary,
        _fallback,
        expected_match_count=2,
        max_latency_seconds=0.5,
    )
    assert out.source == "fallback"
    assert out.used_fallback is True


def test_validate_data_contract_missing_required_raises():
    with pytest.raises(ValueError, match="Missing required contract columns"):
        validate_data_contract(pd.DataFrame({"home_team": ["A"]}))


def test_quality_gates_detect_duplicates_and_invalid_odds():
    df = _contract_df()
    df.loc[1, "match_id"] = 1
    df["home_odds"] = [1.9, 0.9]
    df["draw_odds"] = [3.2, 3.1]
    df["away_odds"] = [4.0, 4.1]
    report = run_quality_gates(df, expected_match_count=2)
    assert report.passed is False
    assert any("duplicate match_id" in x for x in report.issues)
    assert any("invalid home_odds" in x for x in report.issues)


def test_save_and_load_daily_snapshot(tmp_path):
    df = _contract_df()
    paths = save_daily_snapshot(
        df,
        snapshot_dir=str(tmp_path),
        dataset_name="kleague",
        as_of_date="2026-07-22",
    )
    loaded = load_snapshot(paths["csv"])
    assert len(loaded) == 2
    meta = json.loads((tmp_path / "kleague_2026-07-22.metadata.json").read_text())
    assert meta["row_count"] == 2
    assert "home_team" in meta["columns"]
