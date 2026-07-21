import numpy as np
import pytest

from penaltyblog.models import MatchOutcomeModel


def _toy_data():
    X = np.array(
        [
            [1.2, 0.3],
            [0.1, 0.2],
            [-1.2, 0.4],
            [0.8, -0.1],
            [0.0, 0.7],
            [-0.8, -0.6],
            [1.0, 0.2],
            [-1.1, 0.1],
            [0.3, 0.5],
        ]
    )
    y = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1])
    return X, y


def test_match_outcome_model_predict_proba_shape_and_sum():
    X, y = _toy_data()
    model = MatchOutcomeModel.frequentist(regularization=0.1, random_state=7)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_match_outcome_model_supports_bayesian_variant():
    X, y = _toy_data()
    model = MatchOutcomeModel.bayesian_variant(
        regularization=0.1,
        prior_variance=10.0,
        random_state=42,
    )
    model.fit(X, y)

    pred = model.predict(X)
    assert len(pred) == len(y)
    assert set(pred).issubset({"home_win", "draw", "away_win"})


def test_match_outcome_model_calibration_and_params():
    X, y = _toy_data()
    model = MatchOutcomeModel.frequentist(random_state=1)
    model.fit(X, y, feature_names=["f1", "f2"])

    result = model.calibrate_temperature(X, y)
    assert result.temperature > 0

    params = model.get_params()
    assert params["weights"].shape == (2, 3)
    assert params["bias"].shape == (3,)
    assert params["feature_names"] == ["f1", "f2"]


def test_match_outcome_model_raises_when_unfitted():
    model = MatchOutcomeModel()
    with pytest.raises(ValueError, match="not yet fitted"):
        model.predict_proba(np.zeros((2, 2)))
