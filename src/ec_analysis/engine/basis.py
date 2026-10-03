"""Regularized second-order bases with explicit, auditable additive terms."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

FloatArray = npt.NDArray[np.float64]


@typing.final
class PairwiseRidgeRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    def __init__(self, alpha: float = 0.1, maximum_terms: int = 2500) -> None:
        self.alpha = alpha
        self.maximum_terms = maximum_terms

    def fit(self, features: FloatArray, target: FloatArray) -> PairwiseRidgeRegressor:
        feature_count = features.shape[1]
        if feature_count * (feature_count + 3) // 2 > self.maximum_terms:
            raise ValueError("Second-order basis exceeds the configured term limit")
        if self.alpha <= 0:
            raise ValueError("Pairwise Ridge requires positive regularization")
        self.basis_ = PolynomialFeatures(degree=2, include_bias=False)
        expanded_values = self.basis_.fit_transform(features)
        self.regression_ = Ridge(alpha=self.alpha).fit(expanded_values, target)
        self.coef_ = np.asarray(self.regression_.coef_, dtype=float)
        self.intercept_ = float(self.regression_.intercept_)
        self.n_features_in_ = feature_count
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        check_is_fitted(self, ["basis_", "regression_"])
        return np.asarray(self.regression_.predict(self.basis_.transform(features)), dtype=float)

    def explain_terms(self, features: FloatArray) -> FloatArray:
        check_is_fitted(self, ["basis_", "regression_"])
        return np.asarray(self.basis_.transform(features) * self.coef_, dtype=float)

    def factor_groups(self, ownership: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        check_is_fitted(self, ["basis_"])
        return tuple(
            tuple(sorted({ownership[index] for index in np.flatnonzero(power_values)}))
            for power_values in self.basis_.powers_
        )
