"""Fixed GAM policy, native balance, bootstrap cloning and leakage protection."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.stable_gam import CrossFactorRidge
from ec_analysis.schemas.dataset import DatasetLoader, PortfolioDataset


def test_cross_factor_basis_excludes_within_factor_products_and_balances() -> None:
    features = np.random.default_rng(16).normal(size=(100, 4))
    ownership = ("a", "a", "b", "b")
    target = 20 + 2 * features[:, 0] + 3 * features[:, 1] * features[:, 3]
    regressor = CrossFactorRidge(alpha=0.1).fit(features, target, ownership=ownership)
    assert regressor.pair_indices_ == ((0, 2), (0, 3), (1, 2), (1, 3))
    assert regressor.coef_.shape == (8,)
    assert np.allclose(
        regressor.predict(features), regressor.explain_terms(features).sum(axis=1) + regressor.intercept_
    )
    assert regressor.factor_groups(ownership)[4:] == (("a", "b"),) * 4
    with pytest.raises(ValueError, match="maximum_terms"):
        CrossFactorRidge(maximum_terms=7).fit(features, target, ownership=ownership)


def test_pipeline_clone_handles_missing_values_categories_and_reverses() -> None:
    random = np.random.default_rng(32)
    values = random.uniform(size=200)
    records = pd.DataFrame(
        {
            "id": [str(index) for index in range(100)] * 2,
            "period": ["T0"] * 100 + ["T1"] * 100,
            "npv": 10 + values**2,
            "a": values,
            "segment": ["red", "blue"] * 100,
        }
    )
    records.loc[1, "a"] = np.nan
    records.loc[101, "segment"] = "new"
    records.loc[2, "segment"] = None
    dataset = PortfolioDataset(records=records, target_metric="npv", feature_names=("a", "segment"))
    candidate = CandidateFactory(model_policy="fixed_gam").create_candidates(dataset)[0]
    model = typing.cast(Pipeline, candidate.estimator)
    training = dataset.period_records("T0")
    model.fit(training[list(dataset.feature_names)], training.npv)
    cloned = clone(model).fit(training[list(dataset.feature_names)], training.npv)
    assert np.array_equal(
        model.predict(records[list(dataset.feature_names)]), cloned.predict(records[list(dataset.feature_names)])
    )
    first = PipelineAttributor().explain_model("stable_gam", model, dataset, records, "portfolio")
    assert abs(first.summary.numerical_error) < 1e-8
    assert first.summary.observed_change == pytest.approx(
        first.summary.attributed_change + first.summary.unexplained_remainder
    )
    reverse = records.assign(period=records.period.map({"T0": "T1", "T1": "T0"}))
    second = PipelineAttributor().explain_model("stable_gam", model, dataset, reverse, "portfolio")
    assert second.summary.unexplained_remainder == pytest.approx(-first.summary.unexplained_remainder)
    for factors, value in first.contribution_vector().items():
        assert second.contribution_vector()[factors] == pytest.approx(-value)


def test_fixed_family_alpha_selection_ignores_test_targets(tmp_path: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(
        PortfolioGenerator(sample_count=160, scenario="mixed").generate_csv(tmp_path / "data.csv")
    )
    runner = BenchmarkRunner(model_policy="fixed_gam", random_seed=19)
    first = runner.run_experiment(dataset)
    changed = dataset.records.copy()
    changed.loc[changed.id.isin(first.split_manifest.testing_ids), "npv"] += 1000
    second = runner.run_experiment(dataclasses.replace(dataset, records=changed))
    assert set(first.fitted_models) == {"stable_gam"}
    assert first.analysis_state.selected_model == second.analysis_state.selected_model == "stable_gam"
    first_result = first.analysis_state.benchmark_summary[0]
    assert first_result.parameters == second.analysis_state.benchmark_summary[0].parameters
    assert len(first_result.tuning_trials) == 3
    assert first_result.validation_rmse == min(
        trial.validation_rmse for trial in first_result.tuning_trials if trial.validation_rmse is not None
    )
    assert first.analysis_state.final_test != second.analysis_state.final_test


def test_fixed_family_does_not_fall_back_on_unobservable_regime(tmp_path: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(
        PortfolioGenerator(sample_count=180, scenario="regime_shift").generate_csv(tmp_path / "data.csv")
    )
    state = BenchmarkRunner(model_policy="fixed_gam", bootstrap_repeats=10).run(dataset)
    assert state.selected_model == "stable_gam"
    assert state.selection_status != "explanation_validated"
    assert state.confidence_level == "low"
    assert state.test_reliability is not None and state.test_reliability.status == "rejected"
