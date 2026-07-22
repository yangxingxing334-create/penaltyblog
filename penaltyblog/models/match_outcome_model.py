"""Multinomial match-outcome model for 1X2 football predictions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

OUTCOME_LABELS = np.array(["home_win", "draw", "away_win"])
N_CLASSES = 3
EPSILON = 1e-12
MIN_TEMPERATURE = 1e-6


def multiclass_log_loss(
    probs: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> float:
    """Compute numerically stable multiclass log loss."""
    losses = -np.log(probs[np.arange(len(y)), y] + EPSILON)
    if sample_weight is None:
        return float(np.mean(losses))
    return float(np.average(losses, weights=sample_weight))


@dataclass
class CalibrationResult:
    """Result of probability calibration."""

    temperature: float
    loss: float


class MatchOutcomeModel:
    """Multinomial logistic regression model for 1X2 match outcomes.

    Parameters
    ----------
    regularization : float, default=0.1
        L2 regularization strength.
    bayesian : bool, default=False
        If True, uses MAP objective with Gaussian prior.
    prior_variance : float, default=5.0
        Prior variance used when ``bayesian=True``.
    random_state : int, optional
        Random seed for deterministic parameter initialization.
    """

    def __init__(
        self,
        regularization: float = 0.1,
        bayesian: bool = False,
        prior_variance: float = 5.0,
        random_state: Optional[int] = None,
    ):
        self.regularization = float(regularization)
        self.bayesian = bool(bayesian)
        self.prior_variance = float(prior_variance)
        self.random_state = random_state

        self.fitted: bool = False
        self.feature_names_: Optional[list[str]] = None
        self.classes_ = OUTCOME_LABELS.copy()
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._temperature: float = 1.0

    @classmethod
    def frequentist(
        cls, regularization: float = 0.1, random_state: Optional[int] = None
    ) -> "MatchOutcomeModel":
        """Create a frequentist variant."""
        return cls(
            regularization=regularization,
            bayesian=False,
            random_state=random_state,
        )

    @classmethod
    def bayesian_variant(
        cls,
        regularization: float = 0.1,
        prior_variance: float = 5.0,
        random_state: Optional[int] = None,
    ) -> "MatchOutcomeModel":
        """Create a Bayesian MAP variant."""
        return cls(
            regularization=regularization,
            bayesian=True,
            prior_variance=prior_variance,
            random_state=random_state,
        )

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        variant = "bayesian" if self.bayesian else "frequentist"
        return (
            "MatchOutcomeModel("
            f"variant={variant}, regularization={self.regularization}, "
            f"status={status})"
        )

    def _validate_training_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        valid = np.isin(y, np.arange(N_CLASSES))
        if not np.all(valid):
            raise ValueError("y values must be encoded as 0, 1, 2.")

    @staticmethod
    def _split_params(
        params: np.ndarray, n_features: int
    ) -> tuple[np.ndarray, np.ndarray]:
        n_logits = n_features * N_CLASSES
        weights = params[:n_logits].reshape(n_features, N_CLASSES)
        bias = params[n_logits : n_logits + N_CLASSES]
        return weights, bias

    def _objective(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        weights, bias = self._split_params(params, X.shape[1])
        logits = X @ weights + bias
        probs = softmax(logits, axis=1)
        nll = multiclass_log_loss(probs, y, sample_weight=sample_weight)

        if self.bayesian:
            reg = np.sum(weights * weights) / (2.0 * self.prior_variance)
        else:
            reg = 0.5 * self.regularization * np.sum(weights * weights)
        return float(nll + reg)

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[list[str]] = None,
        sample_weight: Optional[Any] = None,
        maxiter: int = 500,
    ) -> "MatchOutcomeModel":
        """Fit the model from tabular features and 1X2 labels."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self._validate_training_inputs(X_arr, y_arr)

        if sample_weight is None:
            w = np.ones(len(y_arr), dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if len(w) != len(y_arr):
                raise ValueError("sample_weight must match y length.")

        self.feature_names_ = feature_names
        rng = np.random.default_rng(self.random_state)
        init = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(X_arr.shape[1] * N_CLASSES + N_CLASSES,),
        )

        res = minimize(
            self._objective,
            x0=init,
            args=(X_arr, y_arr, w),
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        if not res.success:
            raise ValueError(f"Optimization failed: {res.message}")

        self._weights, self._bias = self._split_params(res.x, X_arr.shape[1])
        self.fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted or self._weights is None or self._bias is None:
            raise ValueError("Model is not yet fitted. Call `.fit()` first.")

    def _clamped_temperature(self, value: Optional[float] = None) -> float:
        temp = self._temperature if value is None else float(value)
        return max(temp, MIN_TEMPERATURE)

    def predict_proba(self, X: Any, calibrated: bool = True) -> np.ndarray:
        """Predict home/draw/away probabilities."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")

        logits = X_arr @ self._weights + self._bias
        if calibrated:
            logits = logits / self._clamped_temperature()
        return softmax(logits, axis=1)

    def predict(self, X: Any, calibrated: bool = True) -> np.ndarray:
        """Predict class labels from probabilities."""
        probs = self.predict_proba(X, calibrated=calibrated)
        return self.classes_[np.argmax(probs, axis=1)]

    def calibrate_temperature(self, X_valid: Any, y_valid: Any) -> CalibrationResult:
        """Calibrate probabilities using temperature scaling."""
        self._check_fitted()
        X_arr = np.asarray(X_valid, dtype=float)
        y_arr = np.asarray(y_valid, dtype=int)
        self._validate_training_inputs(X_arr, y_arr)

        logits = X_arr @ self._weights + self._bias

        def loss(temp_arr: np.ndarray) -> float:
            temp = self._clamped_temperature(float(temp_arr[0]))
            probs = softmax(logits / temp, axis=1)
            return multiclass_log_loss(probs, y_arr)

        res = minimize(
            loss, x0=np.array([1.0]), bounds=[(1e-3, 100.0)], method="L-BFGS-B"
        )
        if not res.success:
            raise ValueError(f"Calibration failed: {res.message}")

        self._temperature = float(res.x[0])
        return CalibrationResult(temperature=self._temperature, loss=float(res.fun))

    def get_params(self) -> dict[str, Any]:
        """Return fitted model parameters in a stable structure."""
        self._check_fitted()
        return {
            "weights": self._weights.copy(),
            "bias": self._bias.copy(),
            "temperature": self._temperature,
            "classes": self.classes_.copy(),
            "feature_names": list(self.feature_names_) if self.feature_names_ else None,
            "variant": "bayesian" if self.bayesian else "frequentist",
        }

    def save(self, filepath: str) -> None:
        """Persist model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "MatchOutcomeModel":
        """Load model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
