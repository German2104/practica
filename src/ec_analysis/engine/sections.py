"""Conditional model sections at a training-only reference, never portfolio attribution."""

from __future__ import annotations

import dataclasses
import itertools
import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import AnalysisState


class ModelCurve(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    factor: str
    values: list[float]
    centered_predictions: list[float]


class ModelSurface(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    factors: tuple[str, str]
    first_values: list[float]
    second_values: list[float]
    interaction_contrast: list[list[float]]


class ModelSections(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    method: str = "conditional_training_reference_v1"
    model_name: str
    reference: dict[str, float | str]
    reference_prediction: float
    curves: list[ModelCurve]
    surfaces: list[ModelSurface]
    limitation: str = (
        "Conditional sections at training T0 medians/modes, within 1-99% numeric quantiles. "
        "Not SHAP, portfolio contributions or causal effects. Correlated factors can yield rare combinations."
    )


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class SectionBuilder:
    curve_points: int = 41
    surface_points: int = 17
    maximum_pairs: int = 10

    def build_sections(
        self, estimator: RegressionModel, dataset: PortfolioDataset, training: pd.DataFrame, state: AnalysisState
    ) -> ModelSections:
        reference: dict[str, float | str] = {}
        for factor_name in dataset.feature_names:
            observed = training[factor_name].dropna()
            reference[factor_name] = (
                (float(observed.median()) if len(observed) else 0.0)
                if factor_name in dataset.numeric_features
                else (str(observed.mode().iloc[0]) if len(observed) else "__missing__")
            )
        reference_frame = pd.DataFrame([reference])
        baseline = float(estimator.predict(reference_frame)[0])
        curves = []
        for factor_name in dataset.numeric_features:
            factor_values = self._grid_values(training[factor_name], self.curve_points)
            section_records = pd.DataFrame([reference] * len(factor_values))
            section_records[factor_name] = factor_values
            curves.append(
                ModelCurve(
                    factor=factor_name,
                    values=factor_values.tolist(),
                    centered_predictions=(estimator.predict(section_records) - baseline).tolist(),
                )
            )
        strengths = {tuple(sorted(item.factors)): abs(item.contribution) for item in state.interactions}
        pairs = sorted(
            itertools.combinations(dataset.numeric_features, 2),
            key=lambda factors: (-strengths.get(tuple(sorted(factors)), 0), factors),
        )[: self.maximum_pairs]
        surfaces = [self._build_surface(estimator, training, reference, baseline, factors) for factors in pairs]
        return ModelSections(
            model_name=state.selected_model,
            reference=reference,
            reference_prediction=baseline,
            curves=curves,
            surfaces=surfaces,
        )

    def _grid_values(self, observations: pd.Series[float], count: int) -> np.typing.NDArray[np.float64]:
        finite_values = observations.dropna()
        if finite_values.empty:
            return np.zeros(count)
        return np.linspace(float(finite_values.quantile(0.01)), float(finite_values.quantile(0.99)), count)

    def _build_surface(
        self,
        estimator: RegressionModel,
        training: pd.DataFrame,
        reference: dict[str, float | str],
        baseline: float,
        factors: tuple[str, str],
    ) -> ModelSurface:
        first_values = self._grid_values(training[factors[0]], self.surface_points)
        second_values = self._grid_values(training[factors[1]], self.surface_points)
        grid_records = pd.DataFrame([reference] * (self.surface_points**2))
        grid_records[factors[0]] = np.repeat(first_values, self.surface_points)
        grid_records[factors[1]] = np.tile(second_values, self.surface_points)
        first_records = grid_records.copy()
        first_records[factors[1]] = reference[factors[1]]
        second_records = grid_records.copy()
        second_records[factors[0]] = reference[factors[0]]
        contrast = (
            estimator.predict(grid_records)
            - estimator.predict(first_records)
            - estimator.predict(second_records)
            + baseline
        )
        return ModelSurface(
            factors=factors,
            first_values=first_values.tolist(),
            second_values=second_values.tolist(),
            interaction_contrast=contrast.reshape((self.surface_points, self.surface_points)).tolist(),
        )
