"""Analytic balance, paired signs, feature ownership and native model explanations."""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.ebm_explainer import EbmExplainer
from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.state import AttributionSummary


def test_linear_attribution_matches_known_formula(tmp_path: pathlib.Path) -> None:
    input_path = PortfolioGenerator(sample_count=250, scenario="linear").generate_csv(tmp_path / "linear.csv")
    dataset = DatasetLoader().load_csv(input_path)
    candidate = CandidateFactory().create_candidates(dataset)[1]
    pipeline = typing.cast(Pipeline, candidate.estimator)
    pipeline.set_params(regressor__alpha=0)
    training_records = dataset.period_records("T0")
    pipeline.fit(training_records[list(dataset.feature_names)], training_records[dataset.target_metric])
    outcome = PipelineAttributor().explain_model("ridge", pipeline, dataset, dataset.records, "portfolio")
    coefficients = {"volume": 0.12, "interest_rate": 500, "pd": -600, "lgd": -20, "ltv": 10}
    for contribution in outcome.summary.factor_contributions:
        feature = contribution.factor_name
        expected = coefficients[feature] * (
            dataset.period_records("T1")[feature].sum() - training_records[feature].sum()
        )
        assert contribution.contribution == pytest.approx(expected, abs=0.002)
    assert not outcome.summary.interactions
    assert abs(outcome.summary.unexplained_remainder) < 0.003
    assert outcome.summary.max_local_error < 1e-10


@pytest.mark.parametrize("model_name", ["mean", "ridge", "gam_spline", "xgboost", "ebm"])
def test_model_attribution_balance_pairing_and_period_reversal(portfolio_csv: pathlib.Path, model_name: str) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    candidate = next(
        item
        for item in CandidateFactory(include_optional=True).create_candidates(dataset)
        if item.model_name == model_name
    )
    if candidate.estimator is None:
        pytest.skip(candidate.unavailable_reason or "Optional estimator unavailable")
    pipeline = typing.cast(Pipeline, candidate.estimator)
    if model_name == "ebm":
        pipeline.set_params(regressor__max_rounds=150, regressor__outer_bags=2)
    if model_name == "xgboost":
        pipeline.set_params(regressor__n_estimators=30)
    training_records = dataset.period_records("T0")
    pipeline.fit(training_records[list(dataset.feature_names)], training_records[dataset.target_metric])
    attributor = PipelineAttributor()
    original = attributor.explain_model(model_name, pipeline, dataset, dataset.records, "portfolio")
    reversed_records = dataset.records.assign(period=dataset.records["period"].map({"T0": "T1", "T1": "T0"}))
    reversed_result = attributor.explain_model(model_name, pipeline, dataset, reversed_records, "portfolio")
    for term, value in original.contribution_vector().items():
        assert reversed_result.contribution_vector()[term] == pytest.approx(-value)
    assert original.summary.predicted_change == pytest.approx(
        pipeline.predict(dataset.period_records("T1")[list(dataset.feature_names)]).astype(float).sum()
        - pipeline.predict(training_records[list(dataset.feature_names)]).astype(float).sum(),
        abs=1e-6,
    )
    assert original.summary.observed_change == pytest.approx(
        original.summary.attributed_change + original.summary.unexplained_remainder
    )
    assert abs(original.summary.numerical_error) <= original.summary.numerical_tolerance
    assert original.local_calculations["id"].nunique() == 40
    with pytest.raises(ValueError, match="paired"):
        attributor.explain_model(model_name, pipeline, dataset, dataset.records.iloc[1:], "portfolio")


@pytest.mark.parametrize("model_name", ["ridge", "gam_spline", "xgboost", "ebm"])
def test_onehot_and_spline_terms_are_grouped_to_original_features(portfolio_csv: pathlib.Path, model_name: str) -> None:
    records = pd.read_csv(portfolio_csv)
    records["segment_with_underscore"] = np.where(records["id"] % 2 == 0, "group_a", "group_b")
    records.loc[records.index % 5 == 0, "ltv"] = np.nan
    records.to_csv(portfolio_csv, index=False)
    dataset = DatasetLoader().load_csv(portfolio_csv)
    candidate = next(
        item
        for item in CandidateFactory(include_optional=True).create_candidates(dataset)
        if item.model_name == model_name
    )
    if candidate.estimator is None:
        pytest.skip(candidate.unavailable_reason or "Optional estimator unavailable")
    pipeline = typing.cast(Pipeline, candidate.estimator)
    if model_name == "ebm":
        pipeline.set_params(regressor__max_rounds=100, regressor__outer_bags=2)
    if model_name == "xgboost":
        pipeline.set_params(regressor__n_estimators=20)
    training_records = dataset.period_records("T0")
    pipeline.fit(training_records[list(dataset.feature_names)], training_records[dataset.target_metric])
    outcome = PipelineAttributor().explain_model(model_name, pipeline, dataset, dataset.records, "portfolio")
    assert set(item.factor_name for item in outcome.summary.factor_contributions) == set(dataset.feature_names)
    assert len(set(item.factors for item in outcome.summary.interactions)) == len(outcome.summary.interactions)
    assert all(first != second for first, second in (item.factors for item in outcome.summary.interactions))
    assert all(set(item.factors).issubset(dataset.feature_names) for item in outcome.summary.interactions)
    assert abs(outcome.summary.numerical_error) <= outcome.summary.numerical_tolerance


def test_ebm_nonidentity_link_is_rejected() -> None:
    class UnsupportedLink:
        link_ = "log"
        term_features_: list[tuple[int, ...]] = [(0,)]
        intercept_ = 0.0

        def eval_terms(self, features: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
            return features

    with pytest.raises(ValueError, match="identity"):
        EbmExplainer(model=UnsupportedLink()).explain_local(np.zeros((2, 1)), ("volume",))


def test_duplicate_effects_and_inconsistent_remainder_are_rejected(portfolio_csv: pathlib.Path) -> None:
    experiment = BenchmarkRunner().run_experiment(DatasetLoader().load_csv(portfolio_csv))
    summary = experiment.analysis_state.portfolio_attribution
    assert summary is not None
    payload = summary.model_dump()
    payload["factor_contributions"].append(payload["factor_contributions"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        AttributionSummary.model_validate(payload)
    payload = summary.model_dump()
    payload["unexplained_remainder"] += 1
    with pytest.raises(ValidationError, match="balance"):
        AttributionSummary.model_validate(payload)


def test_attribution_failure_keeps_predictions_and_full_residual(
    portfolio_csv: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PipelineAttributor.explain_model

    def refuse_portfolio(
        self: PipelineAttributor,
        model_name: str,
        estimator: object,
        dataset: object,
        evaluation_records: pd.DataFrame,
        scope: str,
    ) -> object:
        if scope == "portfolio":
            raise ValueError("Simulated additivity failure")
        return original(self, model_name, estimator, dataset, evaluation_records, scope)  # type: ignore[arg-type]

    monkeypatch.setattr(PipelineAttributor, "explain_model", refuse_portfolio)
    state = BenchmarkRunner().run(DatasetLoader().load_csv(portfolio_csv))
    assert state.portfolio_attribution is None
    assert state.unexplained_remainder == pytest.approx(state.total_change)
    assert state.factor_contributions == []
    assert any("Simulated additivity failure" in limitation for limitation in state.limitations)
    assert state.final_test is not None


def test_cancellation_scale_accepts_roundoff_but_rejects_large_error(
    portfolio_csv: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ec_analysis.engine.terms import LocalTerms

    dataset = DatasetLoader().load_csv(portfolio_csv)
    pipeline = typing.cast(Pipeline, CandidateFactory().create_candidates(dataset)[1].estimator)
    training = dataset.period_records("T0")
    pipeline.fit(training[list(dataset.feature_names)], training[dataset.target_metric])
    original_method = PipelineAttributor._model_terms

    def perturbed_terms(self: PipelineAttributor, *arguments: typing.Any) -> LocalTerms:
        original = original_method(self, *arguments)
        values = original.values.copy()
        values[:, 0] += 1e6
        values[:, 1] -= 1e6
        values[:, 0] += 0.002
        return LocalTerms(
            values=values, factors=original.factors, base_value=original.base_value, method=original.method
        )

    monkeypatch.setattr(PipelineAttributor, "_model_terms", perturbed_terms)
    outcome = PipelineAttributor().explain_model("ridge", pipeline, dataset, dataset.records, "portfolio")
    assert outcome.summary.max_local_error == pytest.approx(0.002, abs=1e-8)
    with pytest.raises(ValueError, match="additivity"):
        PipelineAttributor(relative_tolerance=1e-10).explain_model(
            "ridge", pipeline, dataset, dataset.records, "portfolio"
        )
