"""Centered spline main effects and cross-factor tensor products in one ridge fit."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ec_analysis.engine.ownership import FeatureOwnership

FloatArray = npt.NDArray[np.float64]


@typing.final
class CrossFactorRidge(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    def __init__(self, alpha: float = 1.0, maximum_terms: int = 2500, interaction_penalty: float = 1000.0) -> None:
        self.alpha = alpha
        self.maximum_terms = maximum_terms
        self.interaction_penalty = interaction_penalty

    def fit(self, features: FloatArray, target: FloatArray, *, ownership: tuple[str, ...]) -> CrossFactorRidge:
        if features.ndim != 2 or features.shape[1] != len(ownership):
            raise ValueError("Encoded features must have one original factor owner each")
        if self.alpha <= 0 or self.interaction_penalty < 1:
            raise ValueError("Regularization must be positive")
        pair_indices = tuple(
            (left, right)
            for left in range(len(ownership))
            for right in range(left + 1, len(ownership))
            if ownership[left] != ownership[right]
        )
        if len(ownership) + len(pair_indices) > self.maximum_terms:
            raise ValueError("Cross-factor basis exceeds maximum_terms")
        self.ownership_ = ownership
        self.pair_indices_ = pair_indices
        self.n_features_in_ = features.shape[1]
        self.feature_center_ = features.mean(axis=0)
        expanded = self._expand(features)
        self.term_center_ = expanded.mean(axis=0)
        self.penalty_scale_ = np.ones(expanded.shape[1])
        self.penalty_scale_[len(ownership) :] = np.sqrt(self.interaction_penalty)
        self.regression_ = Ridge(alpha=self.alpha).fit((expanded - self.term_center_) / self.penalty_scale_, target)
        self.coef_ = np.asarray(self.regression_.coef_, dtype=float) / self.penalty_scale_
        self.intercept_ = float(self.regression_.intercept_)
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        return np.asarray(self.explain_terms(features).sum(axis=1) + self.intercept_, dtype=float)

    def explain_terms(self, features: FloatArray) -> FloatArray:
        check_is_fitted(self, "regression_")
        return np.asarray((self._expand(features) - self.term_center_) * self.coef_, dtype=float)

    def factor_groups(self, ownership: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        check_is_fitted(self, "regression_")
        if ownership != self.ownership_:
            raise ValueError("Attribution ownership differs from fitted basis")
        return tuple((factor,) for factor in ownership) + tuple(
            tuple(sorted((ownership[left], ownership[right]))) for left, right in self.pair_indices_
        )

    def _expand(self, features: FloatArray) -> FloatArray:
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Encoded feature count changed after fitting")
        centered = features - self.feature_center_
        if not self.pair_indices_:
            return centered
        left, right = zip(*self.pair_indices_, strict=True)
        return np.asarray(np.column_stack((centered, centered[:, left] * centered[:, right])), dtype=float)


@typing.final
class StableGAMPipeline(Pipeline):  # type: ignore[misc]
    """Sklearn pipeline extension: derive ownership from training-only fitted encoders."""

    def __init__(
        self, steps: list[tuple[str, object]], numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        super().__init__(steps)

    def fit(self, features: pd.DataFrame, target: pd.Series[float], **params: object) -> StableGAMPipeline:
        if params:
            raise ValueError("Stable GAM does not accept extra fit parameters")
        preprocessing = self.named_steps["preprocess"]
        transformed = np.asarray(preprocessing.fit_transform(features, target), dtype=float)
        ownership = FeatureOwnership().from_preprocessor(
            preprocessing, self.numeric_features, self.categorical_features
        )
        self.named_steps["regressor"].fit(transformed, np.asarray(target, dtype=float), ownership=ownership)
        return self
