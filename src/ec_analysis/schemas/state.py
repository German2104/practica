"""Auditable result schema shared by analysis and interpretation."""

from __future__ import annotations

import math
import typing

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ec_analysis.schemas.diagnostics import DatasetProfile, GlobalImportance, ParameterSensitivity, ShapeSummary


class FiniteResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class FactorContribution(FiniteResult):
    factor_name: str
    contribution: float
    effect_type: typing.Literal["linear", "threshold", "saturation", "interaction", "nonlinear", "categorical"]
    comment: str = ""
    calculation_ref: str = ""


class InteractionEffect(FiniteResult):
    factors: tuple[str, str]
    contribution: float
    description: str = ""
    calculation_ref: str = ""


class AttributionSummary(FiniteResult):
    model_name: str
    scope: typing.Literal["portfolio", "validation", "test"]
    method: str
    loan_count: int = Field(gt=0)
    factor_contributions: list[FactorContribution]
    interactions: list[InteractionEffect]
    observed_change: float
    predicted_change: float
    attributed_change: float
    prediction_remainder: float
    unexplained_remainder: float
    remainder_fraction: float | None
    remainder_denominator_floor: float = Field(gt=0)
    numerical_error: float
    numerical_tolerance: float = Field(gt=0)
    max_local_error: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    calculation_ref: str

    @model_validator(mode="after")
    def validate_attribution(self) -> AttributionSummary:
        factor_names = [item.factor_name for item in self.factor_contributions]
        interaction_pairs = [tuple(sorted(item.factors)) for item in self.interactions]
        if len(set(factor_names)) != len(factor_names) or len(set(interaction_pairs)) != len(interaction_pairs):
            raise ValueError("Attribution contains duplicate factors or interactions")
        if any(first == second for first, second in interaction_pairs):
            raise ValueError("Within-factor interactions must be grouped into the main contribution")
        reconstructed = sum(item.contribution for item in self.factor_contributions)
        reconstructed += sum(item.contribution for item in self.interactions)
        equations = [
            (self.attributed_change, reconstructed),
            (self.observed_change, self.attributed_change + self.unexplained_remainder),
            (self.prediction_remainder, self.observed_change - self.predicted_change),
            (self.numerical_error, self.predicted_change - self.attributed_change),
        ]
        if any(not math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-6) for first, second in equations):
            raise ValueError("Attribution does not balance predictions or observed change")
        if abs(self.numerical_error) > self.numerical_tolerance:
            raise ValueError("Attribution numerical error exceeds its tolerance")
        expected = (
            self.unexplained_remainder / abs(self.observed_change)
            if abs(self.observed_change) > self.remainder_denominator_floor
            else None
        )
        if expected is None and self.remainder_fraction is not None:
            raise ValueError("Attribution remainder fraction is undefined")
        if expected is not None and (
            self.remainder_fraction is None or not math.isclose(expected, self.remainder_fraction)
        ):
            raise ValueError("Attribution remainder fraction is inconsistent")
        return self


class TermStability(FiniteResult):
    factors: tuple[str, ...]
    reference_contribution: float
    mean_contribution: float
    standard_deviation: float = Field(ge=0)
    percentile_025: float
    percentile_975: float
    sign_agreement: float = Field(ge=0, le=1)
    is_material: bool


class StabilitySummary(FiniteResult):
    method: str = "training_loan_bootstrap_l1_v1"
    scope: typing.Literal["validation"] = "validation"
    requested_repeats: int
    completed_repeats: int
    random_seed: int
    status: typing.Literal["completed", "incomplete", "no_signal"]
    stability_score: float | None = Field(default=None, ge=0, le=1)
    material_sign_agreement: float | None = Field(default=None, ge=0, le=1)
    mean_rank_correlation: float | None = Field(default=None, ge=-1, le=1)
    material_fraction: float = Field(gt=0, lt=1)
    warning_threshold: float
    reference_threshold: float
    term_statistics: list[TermStability] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_seconds: float = Field(ge=0)


class SplitManifest(FiniteResult):
    protocol_version: str = "forward_grouped_v1"
    random_seed: int
    validation_fraction: float
    test_fraction: float
    training_ids: list[str]
    validation_ids: list[str]
    testing_ids: list[str]
    training_period: str = "T0"
    prediction_period: str = "T1"
    calendar_dates_verified: bool
    partition_method: typing.Literal["shuffle", "stable_hash"] = "shuffle"

    @model_validator(mode="after")
    def validate_disjoint(self) -> SplitManifest:
        partitions = [set(self.training_ids), set(self.validation_ids), set(self.testing_ids)]
        if any(not partition for partition in partitions):
            raise ValueError("All split partitions must be nonempty")
        if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
            raise ValueError("Loan partitions must not overlap")
        return self


class EvaluationMetrics(FiniteResult):
    scope: typing.Literal["validation", "test"]
    loan_count: int
    rmse: float = Field(ge=0)
    mae: float = Field(ge=0)
    initial_value: float
    final_value: float
    total_change: float
    predicted_change: float
    signed_remainder: float
    absolute_remainder: float = Field(ge=0)
    remainder_fraction: float | None
    remainder_denominator_floor: float = Field(gt=0)
    normalization_scale: float | None = Field(default=None, gt=0)
    normalized_rmse: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_metrics(self) -> EvaluationMetrics:
        if not math.isclose(self.total_change, self.final_value - self.initial_value, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Evaluation total_change does not match period totals")
        if not math.isclose(
            self.signed_remainder, self.total_change - self.predicted_change, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("Evaluation remainder does not balance predictions")
        if not math.isclose(self.absolute_remainder, abs(self.signed_remainder), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Absolute remainder does not match signed remainder")
        expected_fraction = (
            self.signed_remainder / abs(self.total_change)
            if abs(self.total_change) > self.remainder_denominator_floor
            else None
        )
        if expected_fraction is None:
            if self.remainder_fraction is not None:
                raise ValueError("Remainder fraction is undefined for a small change")
        elif self.remainder_fraction is None or not math.isclose(self.remainder_fraction, expected_fraction):
            raise ValueError("Remainder fraction does not match evaluation metrics")
        if (self.normalization_scale is None) != (self.normalized_rmse is None):
            raise ValueError("Normalized RMSE requires a measured scale")
        if self.normalization_scale is not None and not math.isclose(
            self.rmse / self.normalization_scale, typing.cast(float, self.normalized_rmse), abs_tol=1e-12
        ):
            raise ValueError("Normalized RMSE does not match its scale")
        return self


class TuningTrial(FiniteResult):
    profile: int
    parameters: dict[str, str | int | float | bool]
    status: typing.Literal["completed", "unavailable", "failed"]
    validation_rmse: float | None
    signed_remainder: float | None
    elapsed_seconds: float = Field(ge=0)
    failure_reason: str | None = None


class ReliabilityAssessment(FiniteResult):
    policy_version: str = "prototype_quality_v1"
    scope: typing.Literal["validation", "test"]
    status: typing.Literal["accepted", "rejected", "incomplete"]
    reasons: list[str]
    measurements: dict[str, float | None]
    thresholds: dict[str, float]

    @model_validator(mode="after")
    def validate_decision(self) -> ReliabilityAssessment:
        if self.status == "accepted" and self.reasons:
            raise ValueError("Accepted reliability cannot contain failed criteria")
        if self.status != "accepted" and not self.reasons:
            raise ValueError("A nonaccepted decision requires reasons")
        return self


class BenchmarkResult(FiniteResult):
    model_name: str
    status: typing.Literal["completed", "unavailable", "failed"] = "completed"
    validation_rmse: float | None = Field(default=None, ge=0)
    predicted_change: float | None = None
    signed_remainder: float | None = None
    elapsed_seconds: float = Field(ge=0)
    stability_score: float | None = Field(default=None, ge=0, le=1)
    is_selected: bool = False
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    validation: EvaluationMetrics | None = None
    failure_reason: str | None = None
    attribution: AttributionSummary | None = None
    attribution_failure: str | None = None
    stability: StabilitySummary | None = None
    tuning_trials: list[TuningTrial] = Field(default_factory=list)
    reliability: ReliabilityAssessment | None = None
    parameter_sensitivity: ParameterSensitivity | None = None

    @model_validator(mode="after")
    def validate_status(self) -> BenchmarkResult:
        if self.status == "completed":
            if self.validation_rmse is None or self.predicted_change is None or self.signed_remainder is None:
                raise ValueError("Completed candidate requires measured metrics")
        elif self.is_selected or self.failure_reason is None:
            raise ValueError("Failed/unavailable candidate needs a reason and cannot be selected")
        return self


class AnalysisState(FiniteResult):
    target_metric: str = "npv"
    initial_value: float
    final_value: float
    total_change: float
    selected_model: str
    benchmark_summary: list[BenchmarkResult] = Field(default_factory=list)
    factor_contributions: list[FactorContribution] = Field(default_factory=list)
    interactions: list[InteractionEffect] = Field(default_factory=list)
    unexplained_remainder: float
    confidence_level: typing.Literal["low", "medium", "high"] = "low"
    limitations: list[str] = Field(default_factory=list)
    protocol_version: str | None = None
    selection_reason: str = ""
    selection_status: typing.Literal["prediction_only", "attribution_checked", "explanation_validated"] = (
        "prediction_only"
    )
    final_test: EvaluationMetrics | None = None
    dataset_sha256: str | None = None
    attribution_version: str = "not_computed"
    portfolio_attribution: AttributionSummary | None = None
    test_attribution: AttributionSummary | None = None
    unexplained_fraction: float | None = None
    validation_reliability: ReliabilityAssessment | None = None
    test_reliability: ReliabilityAssessment | None = None
    dataset_profile: DatasetProfile | None = None
    global_importance: GlobalImportance | None = None
    shape_diagnostics: ShapeSummary | None = None

    @model_validator(mode="after")
    def validate_balance(self) -> AnalysisState:
        if not math.isclose(self.total_change, self.final_value - self.initial_value, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("total_change does not match period values")
        contribution_total = sum(item.contribution for item in self.factor_contributions)
        interaction_total = sum(item.contribution for item in self.interactions)
        reconstructed_change = contribution_total + interaction_total + self.unexplained_remainder
        if not math.isclose(self.total_change, reconstructed_change, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Factor contributions and residual do not balance total_change")
        if self.portfolio_attribution is not None:
            summary = self.portfolio_attribution
            if summary.scope != "portfolio" or summary.model_name != self.selected_model:
                raise ValueError("Portfolio attribution does not match the selected model/scope")
            if (
                self.factor_contributions != summary.factor_contributions
                or self.interactions != summary.interactions
                or not math.isclose(self.total_change, summary.observed_change, rel_tol=1e-9, abs_tol=1e-6)
                or self.unexplained_fraction != summary.remainder_fraction
            ):
                raise ValueError("Analysis fields disagree with their checked attribution")
        if self.selection_status == "explanation_validated":
            if (
                self.portfolio_attribution is None
                or self.validation_reliability is None
                or self.test_reliability is None
                or self.validation_reliability.status != "accepted"
                or self.test_reliability.status != "accepted"
            ):
                raise ValueError("Validated explanation requires accepted validation, test and portfolio attribution")
        if self.confidence_level != "low" and self.selection_status != "explanation_validated":
            raise ValueError("Increased confidence requires a validated explanation")
        return self
