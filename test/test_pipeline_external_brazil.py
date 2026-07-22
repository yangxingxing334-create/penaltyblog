import pandas as pd
import pytest
import requests

from penaltyblog.pipelines.data_sources import (
    fetch_external_brazil_serie_a_contract_matches,
    split_historical_and_target_fixtures,
)


class _DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_external_brazil_contract_matches_parses_and_maps(monkeypatch):
    payload = {
        "matches": [
            {
                "num": 1,
                "date": "2026-07-22",
                "team1": "SE Palmeiras",
                "team2": "São Paulo FC",
                "score": {"ft": [2, 0]},
            },
            {
                "num": 2,
                "date": "2026-07-23",
                "team1": "CR Flamengo",
                "team2": "SC Recife",
            },
        ]
    }

    monkeypatch.setattr(
        "penaltyblog.pipelines.data_sources.requests.get",
        lambda *args, **kwargs: _DummyResponse(payload),
    )

    df = fetch_external_brazil_serie_a_contract_matches(
        season=2026,
        team_mappings={
            "Palmeiras": ["SE Palmeiras"],
            "Flamengo": ["CR Flamengo"],
        },
    )
    assert list(df.columns[:6]) == [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    assert "Palmeiras" in df["home_team"].values
    assert "Flamengo" in df["home_team"].values
    assert pd.isna(df.loc[df["match_id"] == 2, "home_goals"]).all()


def test_fetch_external_brazil_contract_matches_http_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("network down")

    monkeypatch.setattr("penaltyblog.pipelines.data_sources.requests.get", _raise)

    with pytest.raises(RuntimeError, match="Temporary external source request failed"):
        fetch_external_brazil_serie_a_contract_matches(season=2026)


def test_split_historical_and_target_fixtures():
    df = pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "match_date": ["2026-07-21", "2026-07-23", "2026-07-23"],
            "home_team": ["A", "C", "E"],
            "away_team": ["B", "D", "F"],
            "home_goals": [1, None, 2],
            "away_goals": [0, None, 2],
        }
    )
    historical, fixtures = split_historical_and_target_fixtures(df, "2026-07-23")
    assert len(historical) == 1
    assert len(fixtures) == 2
    assert fixtures.loc[fixtures["match_id"] == 2, "home_goals"].iloc[0] == 0
    assert fixtures.loc[fixtures["match_id"] == 2, "away_goals"].iloc[0] == 0
