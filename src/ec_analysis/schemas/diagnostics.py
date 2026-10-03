"""Measured diagnostics, distinct from signed portfolio contributions."""

from __future__ import annotations

import typing

from pydantic import BaseModel, ConfigDict, Field


class DiagnosticRecord(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class ColumnProfile(DiagnosticRecord):
    name: str
    kind: typing.Literal["numeric", "categorical"]
    missing_count: int
    unique_count: int
    period_statistics: dict[str, dict[str, float | int | None]]
    category_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CorrelatedFactors(DiagnosticRecord):
    factors: tuple[str, str]
    pearson: float
    spearman: float


class DatasetProfile(DiagnosticRecord):
    method: str = "paired_csv_profile_v1"
    row_count: int
    paired_count: int
    period_counts: dict[str, int]
    target_metric: str
    columns: list[ColumnProfile]
    target_distribution: ColumnProfile | None = None
    correlations: list[CorrelatedFactors]
    calendar_dates_verified: bool
    warnings: list[str]
    calculation_ref: str = "profile.json"


class FactorImportance(DiagnosticRecord):
    factor_name: str
    mean_rmse_increase: float
    standard_deviation: float = Field(ge=0)
    minimum: float
    maximum: float
    repeat_values: list[float]
    calculation_ref: str


class GlobalImportance(DiagnosticRecord):
    method: str = "validation_T1_permutation_rmse_v1"
    scope: typing.Literal["validation"] = "validation"
    sample_count: int
    baseline_rmse: float = Field(ge=0)
    random_seed: int
    repeats: int
    factors: list[FactorImportance]
    warnings: list[str]
    calculation_ref: str = "global-importance.json"


ShapeKind = typing.Literal[
    "linear", "threshold", "saturation", "u_shaped", "inverted_u_shaped", "regime", "complex", "unknown"
]


class ShapeDiagnostic(DiagnosticRecord):
    factor_name: str
    shape: ShapeKind
    variation: float = Field(ge=0)
    relative_linear_error: float = Field(ge=0)
    turning_point: float | None = None
    breakpoint: float | None = None
    tail_slope_ratio: float | None = None
    explanation: str
    calculation_ref: str


class ShapeSummary(DiagnosticRecord):
    method: str = "conditional_section_shape_v1"
    model_name: str
    minimum_variation: float = Field(default=1e-8, ge=0)
    classification_tolerances: dict[str, float] = Field(default_factory=dict)
    factors: list[ShapeDiagnostic]
    warnings: list[str]
    calculation_ref: str = "shapes.json"


class ParameterVariant(DiagnosticRecord):
    profile: int
    validation_rmse: float | None
    status: typing.Literal["completed", "failed"]
    similarity: float | None = None
    sign_agreement: float | None = None
    contributions: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None


class ParameterSensitivity(DiagnosticRecord):
    method: str = "validation_parameter_attribution_v1"
    status: typing.Literal["completed", "unstable", "incomplete", "no_signal"]
    minimum_similarity: float | None
    minimum_sign_agreement: float | None
    similarity_threshold: float = 0.85
    sign_threshold: float = 0.9
    variants: list[ParameterVariant]
    warnings: list[str]
    calculation_ref: str
