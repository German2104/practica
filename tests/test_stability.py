"""Bootstrap resampling boundaries and interpretable stability diagnostics."""

from __future__ import annotations

import pathlib
import typing

import pytest
from sklearn.pipeline import Pipeline

from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.splits import ForwardLoanSplit, select_loans
from ec_analysis.engine.stability import AttributionBootstrap
from ec_analysis.schemas.dataset import DatasetLoader


def test_training_bootstrap_is_reproducible_and_never_uses_validation_targets(tmp_path: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(
        PortfolioGenerator(sample_count=120, scenario="linear").generate_csv(tmp_path / "linear.csv")
    )
    partitions = ForwardLoanSplit().create_split(dataset)
    training_records = select_loans(dataset, partitions.training_ids, "T0")
    validation_records = select_loans(dataset, partitions.validation_ids)
    pipeline = typing.cast(Pipeline, CandidateFactory().create_candidates(dataset)[1].estimator)
    pipeline.set_params(regressor__alpha=0)
    pipeline.fit(training_records[list(dataset.feature_names)], training_records[dataset.target_metric])
    reference = PipelineAttributor().explain_model("ridge", pipeline, dataset, validation_records, "validation")
    evaluator = AttributionBootstrap(requested_repeats=5, random_seed=101)
    first_result = evaluator.evaluate_stability(
        "ridge", pipeline, dataset, training_records, validation_records, reference
    )
    corrupted_validation = validation_records.copy()
    corrupted_validation[dataset.target_metric] += 1e8
    second_result = evaluator.evaluate_stability(
        "ridge", pipeline, dataset, training_records, corrupted_validation, reference
    )
    assert first_result.replicates == second_result.replicates
    assert first_result.summary.stability_score is not None and first_result.summary.stability_score > 0.999
    assert first_result.summary.material_sign_agreement == 1.0
    assert first_result.summary.completed_repeats == 5
    with pytest.raises(ValueError, match="independent"):
        evaluator.evaluate_stability("ridge", pipeline, dataset, training_records, dataset.records, reference)
    with pytest.raises(ValueError, match="T0"):
        evaluator.evaluate_stability("ridge", pipeline, dataset, dataset.records, validation_records, reference)


def test_diagnostics_distinguish_sign_reversal_no_signal_and_incomplete_repeats() -> None:
    evaluator = AttributionBootstrap(requested_repeats=5)
    reference: dict[tuple[str, ...], float] = {("volume",): 100.0, ("ltv",): -20.0}
    stable_result = evaluator._summarize(reference, [reference.copy() for _ in range(5)], [], 0)
    assert stable_result.stability_score == 1.0
    assert stable_result.material_sign_agreement == 1.0
    assert stable_result.mean_rank_correlation == pytest.approx(1.0)
    reversed_vector: dict[tuple[str, ...], float] = {("volume",): -100.0, ("ltv",): 20.0}
    unstable_result = evaluator._summarize(reference, [reversed_vector.copy() for _ in range(5)], [], 0)
    assert unstable_result.stability_score == 0.0
    assert unstable_result.material_sign_agreement == 0.0
    assert any("Unstable" in warning for warning in unstable_result.warnings)
    incomplete = evaluator._summarize(reference, [reference.copy()], ["replicate 1 failed"], 0)
    assert incomplete.status == "incomplete"
    assert incomplete.stability_score is None
    no_signal = evaluator._summarize({("volume",): 0}, [], [], 0)
    assert no_signal.status == "no_signal"
    assert no_signal.stability_score is None
