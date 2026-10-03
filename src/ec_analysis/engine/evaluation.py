"""Compute held-out prediction errors and signed portfolio change errors."""

from __future__ import annotations

import dataclasses
import math
import typing

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import EvaluationMetrics


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EvaluationOutcome:
    metrics: EvaluationMetrics
    predictions: pd.DataFrame


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ModelEvaluator:
    relative_change_floor: float = 1e-6
    absolute_change_floor: float = 1e-8

    def evaluate_model(
        self,
        estimator: RegressionModel,
        dataset: PortfolioDataset,
        evaluation_records: pd.DataFrame,
        scope: typing.Literal["validation", "test"],
    ) -> EvaluationOutcome:
        predictions = evaluation_records[["id", "period", dataset.target_metric]].copy()
        predicted_values = np.asarray(estimator.predict(evaluation_records[list(dataset.feature_names)]), dtype=float)
        if predicted_values.shape != (len(evaluation_records),) or not np.isfinite(predicted_values).all():
            raise ValueError("Candidate produced nonfinite or incorrectly shaped predictions")
        predictions["prediction"] = predicted_values
        initial_records = predictions.loc[predictions["period"] == "T0"]
        final_records = predictions.loc[predictions["period"] == "T1"]
        initial_value = float(initial_records[dataset.target_metric].sum())
        final_value = float(final_records[dataset.target_metric].sum())
        total_change = final_value - initial_value
        predicted_change = float(final_records["prediction"].sum() - initial_records["prediction"].sum())
        signed_remainder = total_change - predicted_change
        change_floor = max(
            self.absolute_change_floor, self.relative_change_floor * max(abs(initial_value), abs(final_value))
        )
        target_values = final_records[dataset.target_metric].to_numpy(dtype=float)
        normalization_scale = max(float(np.std(target_values)), 1e-8 * max(float(np.mean(np.abs(target_values))), 1))
        rmse_value = math.sqrt(mean_squared_error(target_values, final_records["prediction"]))
        metrics = EvaluationMetrics(
            scope=scope,
            loan_count=len(final_records),
            rmse=rmse_value,
            mae=float(mean_absolute_error(final_records[dataset.target_metric], final_records["prediction"])),
            initial_value=initial_value,
            final_value=final_value,
            total_change=total_change,
            predicted_change=predicted_change,
            signed_remainder=signed_remainder,
            absolute_remainder=abs(signed_remainder),
            remainder_denominator_floor=change_floor,
            remainder_fraction=signed_remainder / abs(total_change) if abs(total_change) > change_floor else None,
            normalization_scale=normalization_scale,
            normalized_rmse=rmse_value / normalization_scale,
        )
        return EvaluationOutcome(metrics=metrics, predictions=predictions)
