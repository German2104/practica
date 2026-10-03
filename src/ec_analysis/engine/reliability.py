"""Explicit prototype acceptance criteria, independent of final-test model selection."""

from __future__ import annotations

import dataclasses
import math
import typing

from ec_analysis.schemas.state import AttributionSummary, EvaluationMetrics, ReliabilityAssessment, StabilitySummary


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class QualityPolicy:
    maximum_normalized_rmse: float = 0.15
    maximum_residual_share: float = 0.15
    relative_materiality: float = 0.01
    minimum_stability: float = 0.85
    minimum_sign_agreement: float = 0.90
    minimum_repeats: int = 10

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in dataclasses.astuple(self)):
            raise ValueError("Quality thresholds must be finite")
        if min(self.maximum_normalized_rmse, self.maximum_residual_share, self.relative_materiality) <= 0:
            raise ValueError("Quality error thresholds must be positive")
        if (
            not 0 <= self.minimum_stability <= 1
            or not 0 <= self.minimum_sign_agreement <= 1
            or self.minimum_repeats < 5
        ):
            raise ValueError("Invalid stability acceptance criteria")

    def assess_explanation(
        self,
        metrics: EvaluationMetrics,
        attribution: AttributionSummary | None,
        stability: StabilitySummary | None,
        attribution_failure: str | None = None,
    ) -> ReliabilityAssessment:
        failed_checks: list[str] = []
        missing_checks: list[str] = []
        measurements: dict[str, float | None] = {
            "normalized_rmse": metrics.normalized_rmse,
            "residual_share": None,
            "residual_scale": None,
            "validation_stability_score": stability.stability_score if stability else None,
            "validation_sign_agreement": stability.material_sign_agreement if stability else None,
            "completed_repeats": stability.completed_repeats if stability else None,
        }
        if metrics.normalized_rmse is None:
            missing_checks.append("Normalized prediction error was not measured.")
        elif metrics.normalized_rmse > self.maximum_normalized_rmse:
            failed_checks.append("Prediction error exceeds the configured normalized RMSE limit.")
        if attribution is None:
            if attribution_failure:
                failed_checks.append(f"Attribution failed: {attribution_failure}")
            else:
                missing_checks.append("Checked attribution is unavailable.")
        else:
            if attribution.scope != metrics.scope or not math.isclose(
                attribution.observed_change, metrics.total_change, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ValueError("Attribution and evaluation must describe the same scope and observed change")
            residual_scale = max(
                abs(metrics.total_change),
                self.relative_materiality * max(abs(metrics.initial_value), abs(metrics.final_value)),
                1e-8,
            )
            residual_share = abs(attribution.unexplained_remainder) / residual_scale
            measurements.update(residual_scale=residual_scale, residual_share=residual_share)
            if residual_share > self.maximum_residual_share:
                failed_checks.append("Unexplained remainder exceeds the configured materiality limit.")
        if stability is None or stability.status != "completed" or stability.completed_repeats < self.minimum_repeats:
            missing_checks.append("A complete bootstrap with enough repeats is required.")
        elif stability.stability_score is None or stability.material_sign_agreement is None:
            missing_checks.append("Stability or material-effect sign agreement is undefined.")
        else:
            if stability.stability_score < self.minimum_stability:
                failed_checks.append("Attribution is below the configured stability threshold.")
            if stability.material_sign_agreement < self.minimum_sign_agreement:
                failed_checks.append("Signs of material effects are not sufficiently consistent.")
        return ReliabilityAssessment(
            scope=metrics.scope,
            status="rejected" if failed_checks else "incomplete" if missing_checks else "accepted",
            reasons=failed_checks + missing_checks,
            measurements=measurements,
            thresholds={name: float(value) for name, value in dataclasses.asdict(self).items()},
        )
