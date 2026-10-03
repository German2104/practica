"""Parameter sensitivity on validation only, evaluated before selecting the final model."""

from __future__ import annotations

import dataclasses
import json
import typing

import numpy as np
import pandas as pd

from ec_analysis.engine.attribution import AttributionOutcome, PipelineAttributor
from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.diagnostics import ParameterSensitivity, ParameterVariant
from ec_analysis.schemas.state import TuningTrial


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ParameterStability:
    attributor: PipelineAttributor = dataclasses.field(default_factory=PipelineAttributor)
    similarity_threshold: float = 0.85
    sign_threshold: float = 0.9

    def compare_variants(
        self,
        model_name: str,
        dataset: PortfolioDataset,
        validation_records: pd.DataFrame,
        reference: AttributionOutcome,
        variants: list[tuple[int, RegressionModel]],
        trials: list[TuningTrial],
    ) -> ParameterSensitivity:
        reference_vector = reference.contribution_vector()
        reference_size = sum(abs(value) for value in reference_vector.values())
        variant_results = []
        fitted_variants = dict(variants)
        for trial_record in trials:
            estimator = fitted_variants.get(trial_record.profile)
            if estimator is None:
                variant_results.append(
                    ParameterVariant(
                        profile=trial_record.profile,
                        status="failed",
                        validation_rmse=trial_record.validation_rmse,
                        failure_reason=trial_record.failure_reason or "Profile did not complete training.",
                    )
                )
                continue
            try:
                outcome = self.attributor.explain_model(
                    model_name, estimator, dataset, validation_records, "validation"
                )
                variant_vector = outcome.contribution_vector()
                similarity, agreement = self._compare_vectors(reference_vector, variant_vector)
                variant_results.append(
                    ParameterVariant(
                        profile=trial_record.profile,
                        status="completed",
                        validation_rmse=trial_record.validation_rmse,
                        similarity=similarity,
                        sign_agreement=agreement,
                        contributions={json.dumps(factors): value for factors, value in variant_vector.items()},
                    )
                )
            except (ValueError, RuntimeError, ArithmeticError, ImportError, OSError) as failure:
                variant_results.append(
                    ParameterVariant(
                        profile=trial_record.profile,
                        status="failed",
                        validation_rmse=trial_record.validation_rmse,
                        failure_reason=f"{type(failure).__name__}: {failure}",
                    )
                )
        completed_results = [item for item in variant_results if item.status == "completed"]
        similarities = [item.similarity for item in completed_results if item.similarity is not None]
        agreements = [item.sign_agreement for item in completed_results if item.sign_agreement is not None]
        minimum_similarity = min(similarities) if similarities else None
        minimum_agreement = min(agreements) if agreements else None
        status: typing.Literal["completed", "unstable", "incomplete", "no_signal"] = "completed"
        if reference_size <= 1e-8:
            status = "no_signal"
        elif len(completed_results) != len(trials) or len(trials) < 2 or minimum_similarity is None:
            status = "incomplete"
        elif minimum_similarity < self.similarity_threshold or (
            minimum_agreement is None or minimum_agreement < self.sign_threshold
        ):
            status = "unstable"
        return ParameterSensitivity(
            status=status,
            minimum_similarity=minimum_similarity,
            minimum_sign_agreement=minimum_agreement,
            similarity_threshold=self.similarity_threshold,
            sign_threshold=self.sign_threshold,
            variants=variant_results,
            calculation_ref=f"parameter-sensitivity-{model_name}.json",
            warnings=[
                "Sensitivity over recorded parameter profiles on fixed validation clients; not all possible settings.",
                "No test targets or generating formulas are used; unstable profiles remain in the result.",
            ],
        )

    def _compare_vectors(
        self, reference: dict[tuple[str, ...], float], alternative: dict[tuple[str, ...], float]
    ) -> tuple[float | None, float | None]:
        factor_keys = sorted(set(reference) | set(alternative))
        reference_values = np.asarray([reference.get(factors, 0) for factors in factor_keys])
        alternative_values = np.asarray([alternative.get(factors, 0) for factors in factor_keys])
        reference_size = float(np.abs(reference_values).sum())
        if reference_size <= 1e-8:
            return None, None
        similarity = 1 - float(np.abs(reference_values - alternative_values).sum()) / (
            reference_size + float(np.abs(alternative_values).sum())
        )
        material_mask = np.abs(reference_values) >= 0.01 * reference_size
        sign_agreement = float(
            np.mean(
                np.sign(reference_values[material_mask])
                == np.where(
                    np.abs(alternative_values[material_mask]) >= 0.01 * reference_size,
                    np.sign(alternative_values[material_mask]),
                    0,
                )
            )
        )
        return float(np.clip(similarity, 0, 1)), sign_agreement
