"""Parameter selection, polynomial attribution and stable multi-window partitions."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.basis import PairwiseRidgeRegressor
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.splits import ForwardLoanSplit, select_loans
from ec_analysis.schemas.dataset import DatasetLoader, PortfolioDataset


def test_quadratic_attribution_recovers_known_interaction_and_reverses() -> None:
    random_source = np.random.default_rng(123)
    initial_values = random_source.normal(size=(200, 2))
    final_values = initial_values + random_source.normal(scale=0.4, size=(200, 2))
    features = np.concatenate([initial_values, final_values])
    records = pd.DataFrame(features, columns=["factor_a", "factor_b"])
    records["id"] = [str(index) for index in range(200)] * 2
    records["period"] = ["T0"] * 200 + ["T1"] * 200
    records["npv"] = 10 + 2 * features[:, 0] + 3 * features[:, 1] ** 2 + 4 * features[:, 0] * features[:, 1]
    dataset = PortfolioDataset(records=records, target_metric="npv", feature_names=("factor_a", "factor_b"))
    candidate = next(
        item
        for item in CandidateFactory(include_extended=True).create_candidates(dataset)
        if item.model_name == "quadratic_ridge"
    )
    estimator = typing.cast(Pipeline, candidate.estimator)
    estimator.set_params(regressor__alpha=1e-8)
    training = dataset.period_records("T0")
    estimator.fit(training[list(dataset.feature_names)], training["npv"])
    original = PipelineAttributor().explain_model("quadratic_ridge", estimator, dataset, records, "portfolio")
    assert len(original.summary.interactions) == 1
    centered_initial = initial_values - initial_values.mean(axis=0)
    centered_final = final_values - initial_values.mean(axis=0)
    expected_pair = 4 * (np.prod(centered_final, axis=1).sum() - np.prod(centered_initial, axis=1).sum())
    assert original.summary.interactions[0].contribution == pytest.approx(expected_pair, abs=1e-5)
    assert abs(original.summary.unexplained_remainder) < 1e-5
    reversed_records = records.assign(period=records["period"].map({"T0": "T1", "T1": "T0"}))
    reversed_result = PipelineAttributor().explain_model(
        "quadratic_ridge", estimator, dataset, reversed_records, "portfolio"
    )
    for factors, contribution in original.contribution_vector().items():
        assert reversed_result.contribution_vector()[factors] == pytest.approx(-contribution)


def test_tuning_chooses_validation_minimum_and_ignores_test_answers(tmp_path: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(
        PortfolioGenerator(sample_count=100, scenario="mixed").generate_csv(tmp_path / "mixed.csv")
    )
    runner = BenchmarkRunner(tune_parameters=True)
    first_result = runner.run_experiment(dataset)
    changed_records = dataset.records.copy()
    changed_records.loc[changed_records["id"].isin(first_result.split_manifest.testing_ids), "npv"] += 1000
    second_result = runner.run_experiment(dataclasses.replace(dataset, records=changed_records))
    assert first_result.analysis_state.selected_model == second_result.analysis_state.selected_model
    for first_model, second_model in zip(
        first_result.analysis_state.benchmark_summary, second_result.analysis_state.benchmark_summary, strict=True
    ):
        expected_trials = 1 if first_model.model_name == "mean" else 3
        assert len(first_model.tuning_trials) == expected_trials
        scores = [trial.validation_rmse for trial in first_model.tuning_trials if trial.validation_rmse is not None]
        assert first_model.validation_rmse == min(scores)
        assert first_model.parameters == second_model.parameters
        assert first_model.validation_rmse == second_model.validation_rmse
    assert first_result.analysis_state.final_test != second_result.analysis_state.final_test


def test_hash_split_keeps_membership_when_window_population_changes(tmp_path: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(PortfolioGenerator(sample_count=200).generate_csv(tmp_path / "portfolio.csv"))
    split_policy = ForwardLoanSplit(partition_method="stable_hash")
    first_split = split_policy.create_split(dataset)
    kept_ids = dataset.records["id"].unique()[15:]
    changed = dataclasses.replace(dataset, records=dataset.records.loc[dataset.records["id"].isin(kept_ids)])
    second_split = split_policy.create_split(changed)
    for partition_name in ("training_ids", "validation_ids", "testing_ids"):
        assert set(getattr(first_split, partition_name)) & set(kept_ids) == set(getattr(second_split, partition_name))
    assert set(select_loans(dataset, first_split.training_ids, "T0")["period"]) == {"T0"}


def test_pairwise_basis_rejects_excessive_size_before_expansion() -> None:
    with pytest.raises(ValueError, match="term limit"):
        PairwiseRidgeRegressor().fit(np.ones((10, 100)), np.ones(10))
