"""Acceptance, refusal, materiality and test-independent selection."""

from __future__ import annotations

import pathlib

import pytest

from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.reliability import QualityPolicy
from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.state import AttributionSummary, EvaluationMetrics, FactorContribution, StabilitySummary


@pytest.fixture
def quality_measurements() -> tuple[EvaluationMetrics, AttributionSummary, StabilitySummary]:
    metrics = EvaluationMetrics(
        scope="validation",
        loan_count=10,
        rmse=1,
        mae=0.8,
        initial_value=100,
        final_value=110,
        total_change=10,
        predicted_change=9,
        signed_remainder=1,
        absolute_remainder=1,
        remainder_fraction=0.1,
        remainder_denominator_floor=1e-4,
        normalization_scale=20,
        normalized_rmse=0.05,
    )
    attribution = AttributionSummary(
        model_name="ridge",
        scope="validation",
        method="test_control",
        loan_count=10,
        factor_contributions=[FactorContribution(factor_name="factor", contribution=9, effect_type="linear")],
        interactions=[],
        observed_change=10,
        predicted_change=9,
        attributed_change=9,
        prediction_remainder=1,
        unexplained_remainder=1,
        remainder_fraction=0.1,
        remainder_denominator_floor=1e-4,
        numerical_error=0,
        numerical_tolerance=1e-4,
        max_local_error=0,
        elapsed_seconds=0,
        calculation_ref="control.csv",
    )
    stability = StabilitySummary(
        requested_repeats=10,
        completed_repeats=10,
        random_seed=42,
        status="completed",
        stability_score=0.95,
        material_sign_agreement=1,
        material_fraction=0.01,
        warning_threshold=0.7,
        reference_threshold=0.85,
        elapsed_seconds=0,
    )
    return metrics, attribution, stability


def test_good_metrics_pass_and_missing_bootstrap_never_means_success(
    quality_measurements: tuple[EvaluationMetrics, AttributionSummary, StabilitySummary],
) -> None:
    metrics, attribution, stability = quality_measurements
    policy = QualityPolicy()
    assert policy.assess_explanation(metrics, attribution, stability).status == "accepted"
    assert policy.assess_explanation(metrics, attribution, None).status == "incomplete"
    assert policy.assess_explanation(metrics, None, stability, "corrupt balance").status == "rejected"
    with pytest.raises(ValueError, match="same scope"):
        policy.assess_explanation(metrics, attribution.model_copy(update={"scope": "test"}), stability)


@pytest.mark.parametrize("failed_check", ["prediction", "residual", "stability", "sign"])
def test_each_failed_criterion_is_reported(
    quality_measurements: tuple[EvaluationMetrics, AttributionSummary, StabilitySummary], failed_check: str
) -> None:
    metrics, attribution, stability = quality_measurements
    if failed_check == "prediction":
        metrics = metrics.model_copy(update={"rmse": 20, "normalized_rmse": 1})
    elif failed_check == "residual":
        payload = attribution.model_dump()
        payload.update(
            attributed_change=1,
            predicted_change=1,
            prediction_remainder=9,
            unexplained_remainder=9,
            remainder_fraction=0.9,
        )
        payload["factor_contributions"][0]["contribution"] = 1
        attribution = AttributionSummary.model_validate(payload)
    elif failed_check == "stability":
        stability = stability.model_copy(update={"stability_score": 0.5})
    else:
        stability = stability.model_copy(update={"material_sign_agreement": 0.5})
    result = QualityPolicy().assess_explanation(metrics, attribution, stability)
    assert result.status == "rejected"
    assert result.reasons


def test_materiality_floor_is_explicit_for_small_changes(
    quality_measurements: tuple[EvaluationMetrics, AttributionSummary, StabilitySummary],
) -> None:
    metrics, attribution, stability = quality_measurements
    metrics = metrics.model_copy(update={"initial_value": 10000, "final_value": 10010})
    result = QualityPolicy().assess_explanation(metrics, attribution, stability)
    assert result.measurements["residual_scale"] == pytest.approx(100.1)
    assert result.measurements["residual_share"] == pytest.approx(1 / 100.1)


def test_linear_explanation_passes_and_hidden_regime_is_rejected(tmp_path: pathlib.Path) -> None:
    for scenario_name in ("linear", "regime_shift"):
        source_path = PortfolioGenerator(sample_count=160, scenario=scenario_name).generate_csv(
            tmp_path / f"{scenario_name}.csv"
        )
        state = BenchmarkRunner(tune_parameters=True, bootstrap_repeats=10).run(DatasetLoader().load_csv(source_path))
        assert state.validation_reliability is not None and state.test_reliability is not None
        if scenario_name == "linear":
            assert state.selection_status == "explanation_validated"
            assert state.confidence_level == "medium"
        else:
            assert state.validation_reliability.status == "rejected"
            assert state.test_reliability.status == "rejected"
            assert state.confidence_level == "low"
