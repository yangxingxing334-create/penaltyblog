import pandas as pd

from penaltyblog.matchflow import Flow
from penaltyblog.pipelines.statsbomb_fetcher import fetch_statsbomb_matches


class _FakeFlow:
    def __init__(self, df):
        self._df = df

    def flatten(self):
        return self

    def to_pandas(self):
        return self._df.copy()


class _FakeStatsBomb:
    def __init__(self, df):
        self._df = df

    def matches(self, competition_id, season_id, creds=None):
        return _FakeFlow(self._df)


def test_fetch_statsbomb_matches_normalizes_and_filters(monkeypatch):
    raw = pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "match_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "home_team.home_team_name": ["A", "B", "C"],
            "away_team.away_team_name": ["B", "C", "A"],
            "home_score": [1, 0, 2],
            "away_score": [0, 0, 1],
        }
    )
    monkeypatch.setattr(Flow, "statsbomb", _FakeStatsBomb(raw), raising=False)

    out = fetch_statsbomb_matches(2, 44, start_date="2023-02-01", end_date="2023-03-01")
    assert list(out["match_id"]) == [2, 3]
    assert {"home_team", "away_team", "home_goals", "away_goals"}.issubset(out.columns)
