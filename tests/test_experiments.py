"""Leakage, reproducibility, diagnostics and persistence of actual ML experiments."""

from __future__ import annotations

import dataclasses
import importlib
import json
import pathlib
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.engine.artifacts import ExperimentWriter
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.evaluation import ModelEvaluator
from ec_analysis.engine.splits import ForwardLoanSplit, select_loans
from ec_analysis.schemas.dataset import DatasetLoader, PortfolioDataset
from ec_analysis.schemas.state import EvaluationMetrics
from scripts.run_benchmark import main


def test_disjoint_split_and_training_preprocessing_use_only_t0(portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    experiment = BenchmarkRunner().run_experiment(dataset)
    partitions = experiment.split_manifest
    assert not set(partitions.training_ids) & set(partitions.validation_ids)
    assert not set(partitions.training_ids) & set(partitions.testing_ids)
    assert not set(partitions.validation_ids) & set(partitions.testing_ids)
    assert len(partitions.training_ids) + len(partitions.validation_ids) + len(partitions.testing_ids) == 40
    estimator = experiment.fitted_models["ridge"]
    preprocessing = estimator.named_steps["preprocess"]  # type: ignore[attr-defined]
    scaling = preprocessing.named_transformers_["numeric"].named_steps["scale"]
    training_records = select_loans(dataset, partitions.training_ids, "T0")
    assert scaling.mean_ == pytest.approx(training_records[list(dataset.numeric_features)].mean().to_numpy())
    reversed_dataset = dataclasses.replace(dataset, records=dataset.records.iloc[::-1])
    assert ForwardLoanSplit().create_split(reversed_dataset) == partitions


def test_test_targets_do_not_change_selection_or_validation(portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    first_run = BenchmarkRunner().run_experiment(dataset)
    dataset.records.loc[dataset.records["id"].isin(first_run.split_manifest.testing_ids), "npv"] += 10000
    second_run = BenchmarkRunner().run_experiment(dataset)
    assert first_run.analysis_state.selected_model == second_run.analysis_state.selected_model
    for first_result, second_result in zip(
        first_run.analysis_state.benchmark_summary, second_run.analysis_state.benchmark_summary, strict=True
    ):
        assert first_result.validation == second_result.validation
    assert first_run.analysis_state.final_test is not None
    assert second_run.analysis_state.final_test is not None
    assert second_run.analysis_state.final_test.rmse > first_run.analysis_state.final_test.rmse + 9000


def test_validation_portfolio_metrics_match_saved_heldout_predictions(portfolio_csv: pathlib.Path) -> None:
    experiment = BenchmarkRunner().run_experiment(DatasetLoader().load_csv(portfolio_csv))
    for result in experiment.analysis_state.benchmark_summary:
        assert result.validation is not None
        predictions = experiment.predictions[f"validation-{result.model_name}"]
        assert set(predictions["id"]) == set(experiment.split_manifest.validation_ids)
        initial_records = predictions.loc[predictions["period"] == "T0"]
        final_records = predictions.loc[predictions["period"] == "T1"]
        actual_delta = final_records["npv"].sum() - initial_records["npv"].sum()
        predicted_delta = final_records["prediction"].sum() - initial_records["prediction"].sum()
        assert result.signed_remainder == pytest.approx(actual_delta - predicted_delta)
        assert result.validation.mae == pytest.approx(abs(final_records["npv"] - final_records["prediction"]).mean())
        assert result.validation.loan_count == len(experiment.split_manifest.validation_ids)
    assert sum(name.startswith("test-") for name in experiment.predictions) == 1


def test_missing_numeric_and_unknown_category_are_processed_from_training_only(portfolio_csv: pathlib.Path) -> None:
    records = pd.read_csv(portfolio_csv)
    records["segment"] = np.where(records["period"] == "T0", "known", "unseen")
    records.loc[records.index % 5 == 0, "ltv"] = np.nan
    records.to_csv(portfolio_csv, index=False)
    dataset = DatasetLoader().load_csv(portfolio_csv)
    experiment = BenchmarkRunner().run_experiment(dataset)
    assert "segment" in dataset.categorical_features
    assert all(result.status == "completed" for result in experiment.analysis_state.benchmark_summary)
    estimator = experiment.fitted_models["ridge"]
    preprocessing = estimator.named_steps["preprocess"]  # type: ignore[attr-defined]
    encoder = preprocessing.named_transformers_["categorical"].named_steps["encode"]
    assert encoder.categories_[0].tolist() == ["known"]
    imputer = preprocessing.named_transformers_["numeric"].named_steps["impute"]
    training_records = select_loans(dataset, experiment.split_manifest.training_ids, "T0")
    assert imputer.statistics_ == pytest.approx(training_records[list(dataset.numeric_features)].median().to_numpy())


@pytest.mark.parametrize("column_name", ["npv", "volume"])
def test_loader_rejects_infinity(portfolio_csv: pathlib.Path, column_name: str) -> None:
    records = pd.read_csv(portfolio_csv)
    records.loc[0, column_name] = np.inf
    records.to_csv(portfolio_csv, index=False)
    with pytest.raises(ValueError, match="finite|infinite"):
        DatasetLoader().load_csv(portfolio_csv)


def test_feature_allowlist_and_calendar_validation(portfolio_csv: pathlib.Path) -> None:
    records = pd.read_csv(portfolio_csv)
    records["as_of_date"] = np.where(records["period"] == "T0", "2026-01-01", "2026-02-01")
    records.to_csv(portfolio_csv, index=False)
    dataset = DatasetLoader(requested_features=("volume", "ltv")).load_csv(portfolio_csv)
    assert dataset.feature_names == ("volume", "ltv")
    assert ForwardLoanSplit().create_split(dataset).calendar_dates_verified
    with pytest.raises(ValueError, match="Factors cannot"):
        DatasetLoader(requested_features=("npv",)).load_csv(portfolio_csv)
    records["as_of_date"] = "2026-01-01"
    records.to_csv(portfolio_csv, index=False)
    with pytest.raises(ValueError, match="precede"):
        DatasetLoader().load_csv(portfolio_csv)


def test_missing_optional_model_is_reported(monkeypatch: pytest.MonkeyPatch, portfolio_csv: pathlib.Path) -> None:
    original_import = importlib.import_module

    def import_without_xgboost(name: str, package: str | None = None) -> types.ModuleType:
        if name in {"xgboost", "interpret.glassbox"}:
            raise ImportError("Deliberately missing optional package")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_xgboost)
    experiment = BenchmarkRunner(include_optional=True).run_experiment(DatasetLoader().load_csv(portfolio_csv))
    results = {result.model_name: result for result in experiment.analysis_state.benchmark_summary}
    assert results["xgboost"].status == results["ebm"].status == "unavailable"
    assert results["xgboost"].failure_reason is not None
    assert results["xgboost"].validation_rmse is None
    assert not results["xgboost"].is_selected


def test_zero_delta_has_no_residual_fraction(portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    dataset.records["npv"] = 100.0
    candidate = CandidateFactory().create_candidates(dataset)[0]
    assert candidate.estimator is not None
    candidate.estimator.fit(dataset.records[list(dataset.feature_names)], dataset.records["npv"])
    outcome = ModelEvaluator().evaluate_model(candidate.estimator, dataset, dataset.records, "validation")
    assert outcome.metrics.total_change == 0
    assert outcome.metrics.remainder_fraction is None
    assert EvaluationMetrics.model_validate_json(outcome.metrics.model_dump_json()) == outcome.metrics


def test_cli_saves_reloadable_models_and_provenance(
    tmp_path: pathlib.Path,
    portfolio_csv: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", str(portfolio_csv), "--output-dir", str(output_directory)])
    assert main() == 0
    manifest = json.loads((output_directory / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == file_checksum(portfolio_csv)
    for filename, expected in manifest["artifact_sha256"].items():
        assert file_checksum(output_directory / filename) == expected
    estimator = joblib.load(output_directory / "model-ridge.joblib")
    dataset = DatasetLoader().load_csv(portfolio_csv)
    saved_predictions = pd.read_csv(output_directory / "validation-ridge.csv", dtype={"id": str})
    ordered_records = dataset.records.merge(saved_predictions[["id", "period"]], on=["id", "period"]).sort_values(
        ["id", "period"]
    )
    assert estimator.predict(ordered_records[list(dataset.feature_names)]) == pytest.approx(
        saved_predictions["prediction"].to_numpy()
    )
    original_manifest = (output_directory / "manifest.json").read_bytes()
    assert main() == 2
    assert (output_directory / "manifest.json").read_bytes() == original_manifest


def test_changed_dataset_cannot_be_saved_as_original(tmp_path: pathlib.Path, portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    original_checksum = file_checksum(portfolio_csv)
    experiment = BenchmarkRunner().run_experiment(dataset)
    portfolio_csv.write_text(portfolio_csv.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        ExperimentWriter(project_root=pathlib.Path.cwd()).save_experiment(
            experiment, dataset, portfolio_csv, original_checksum, tmp_path / "run"
        )
    assert not (tmp_path / "run").exists()


def test_failed_candidate_has_no_metrics_or_selection(
    monkeypatch: pytest.MonkeyPatch,
    portfolio_csv: pathlib.Path,
) -> None:
    from ec_analysis.engine.candidates import ModelCandidate

    original_factory = CandidateFactory.create_candidates

    class InvalidRegressor:
        def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> object:
            return self

        def predict(self, features: pd.DataFrame) -> np.typing.NDArray[np.float64]:
            return np.full(len(features), np.nan)

    def candidates_with_failure(factory: CandidateFactory, dataset: PortfolioDataset) -> list[ModelCandidate]:
        return original_factory(factory, dataset) + [
            ModelCandidate(model_name="nonfinite", estimator=InvalidRegressor(), parameters={})
        ]

    monkeypatch.setattr(CandidateFactory, "create_candidates", candidates_with_failure)
    experiment = BenchmarkRunner().run_experiment(DatasetLoader().load_csv(portfolio_csv))
    failed_result = experiment.analysis_state.benchmark_summary[-1]
    assert failed_result.status == "failed"
    assert failed_result.failure_reason is not None
    assert failed_result.validation is None
    assert not failed_result.is_selected
