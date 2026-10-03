"""Train comparable models, select on validation, then evaluate the frozen winner."""

from __future__ import annotations

import dataclasses
import typing

import pandas as pd

from ec_analysis.engine.candidates import ModelPolicy, RegressionModel
from ec_analysis.engine.diagnostics import DiagnosticRunner
from ec_analysis.engine.evaluation import ModelEvaluator
from ec_analysis.engine.explanation import ExplanationRunner
from ec_analysis.engine.reliability import QualityPolicy
from ec_analysis.engine.search import ModelSearch
from ec_analysis.engine.splits import ForwardLoanSplit, select_loans
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import AnalysisState, SplitManifest


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class BenchmarkExperiment:
    analysis_state: AnalysisState
    split_manifest: SplitManifest
    fitted_models: dict[str, RegressionModel]
    predictions: dict[str, pd.DataFrame]
    attribution_tables: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    bootstrap_tables: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class BenchmarkRunner:
    model_policy: ModelPolicy = "benchmark"
    test_fraction: float = 0.2
    random_seed: int = 42
    include_optional: bool = False
    validation_fraction: float = 0.2
    evaluator: ModelEvaluator = dataclasses.field(default_factory=ModelEvaluator)
    include_attribution: bool = True
    bootstrap_repeats: int = 0
    tune_parameters: bool = False
    worker_count: int = 2
    partition_method: typing.Literal["shuffle", "stable_hash"] = "shuffle"
    quality_policy: QualityPolicy = dataclasses.field(default_factory=QualityPolicy)

    def run(self, dataset: PortfolioDataset) -> AnalysisState:
        return self.run_experiment(dataset).analysis_state

    def run_experiment(self, dataset: PortfolioDataset) -> BenchmarkExperiment:
        if self.bootstrap_repeats and (not self.include_attribution or self.bootstrap_repeats < 5):
            raise ValueError("Bootstrap requires attribution and at least five repeats")
        split_manifest = ForwardLoanSplit(
            random_seed=self.random_seed,
            test_fraction=self.test_fraction,
            validation_fraction=self.validation_fraction,
            partition_method=self.partition_method,
        ).create_split(dataset)
        training_records = select_loans(dataset, split_manifest.training_ids, "T0")
        validation_records = select_loans(dataset, split_manifest.validation_ids)
        search_result = ModelSearch(
            model_policy=self.model_policy,
            random_seed=self.random_seed,
            include_optional=self.include_optional,
            tune_parameters=self.tune_parameters,
            worker_count=self.worker_count,
            evaluator=self.evaluator,
        ).fit_candidates(dataset, training_records, validation_records)
        explanation_runner = ExplanationRunner(
            bootstrap_repeats=self.bootstrap_repeats,
            random_seed=self.random_seed,
            quality_policy=self.quality_policy,
        )
        attribution_tables: dict[str, pd.DataFrame] = {}
        bootstrap_tables: dict[str, pd.DataFrame] = {}
        if self.include_attribution:
            validation_diagnostics = explanation_runner.evaluate_validation(search_result, dataset, split_manifest)
            search_result = validation_diagnostics.search_result
            attribution_tables = validation_diagnostics.attribution_tables
            bootstrap_tables = validation_diagnostics.bootstrap_tables
        benchmark_results = search_result.benchmark_results
        fitted_models = search_result.fitted_models
        predictions = search_result.predictions
        completed_results = [result for result in benchmark_results if result.status == "completed"]
        if not completed_results:
            raise ValueError("No candidate completed; reliable prediction or explanation is unavailable")
        accepted_results = [
            result
            for result in completed_results
            if result.reliability is not None and result.reliability.status == "accepted"
        ]
        selected_result = min(
            accepted_results or completed_results,
            key=lambda result: (typing.cast(float, result.validation_rmse), result.model_name),
        )
        selected_result.is_selected = True
        selection_reason = (
            "Fixed GAM family; alpha has lowest validation RMSE among three predefined profiles. "
            "No fallback family. Test data do not select parameters."
            if self.model_policy == "fixed_gam"
            else (
                "Lowest validation RMSE among candidates passing all validation quality criteria."
                if accepted_results
                else "No accepted explanation; lowest validation RMSE is retained for diagnostics."
            )
            + " Ties by model name. Test data do not select the model."
        )
        # Test targets are first evaluated after the candidate has been frozen; there is no refit.
        test_outcome = self.evaluator.evaluate_model(
            fitted_models[selected_result.model_name],
            dataset,
            select_loans(dataset, split_manifest.testing_ids),
            "test",
        )
        predictions[f"test-{selected_result.model_name}"] = test_outcome.predictions
        initial_value = float(dataset.period_records("T0")[dataset.target_metric].sum())
        final_value = float(dataset.period_records("T1")[dataset.target_metric].sum())
        limitations = [
            "Prediction-only selection: factor contributions and stability have not been evaluated.",
            "State totals describe the full input portfolio; candidate and test errors describe held-out loans only.",
            "One T0/T1 transition on disjoint loans is not a multi-window temporal reliability assessment.",
            "Parameter search uses validation only; predictive reliability is not yet certified."
            if self.tune_parameters or self.model_policy == "fixed_gam"
            else "Parameters are fixed; no search or predictive reliability gate was performed.",
        ]
        if not split_manifest.calendar_dates_verified:
            limitations.append("Calendar dates are absent; only the declared relative order T0 before T1 is used.")
        analysis_state = AnalysisState(
            target_metric=dataset.target_metric,
            initial_value=initial_value,
            final_value=final_value,
            total_change=final_value - initial_value,
            selected_model=selected_result.model_name,
            benchmark_summary=benchmark_results,
            unexplained_remainder=final_value - initial_value,
            protocol_version=split_manifest.protocol_version,
            final_test=test_outcome.metrics,
            selection_reason=selection_reason,
            limitations=limitations,
        )
        experiment = BenchmarkExperiment(
            analysis_state=analysis_state,
            split_manifest=split_manifest,
            fitted_models=fitted_models,
            predictions=predictions,
            attribution_tables=attribution_tables,
            bootstrap_tables=bootstrap_tables,
        )
        if self.include_attribution:
            experiment = explanation_runner.enrich_experiment(experiment, dataset)
        return DiagnosticRunner(random_seed=self.random_seed).enrich_experiment(experiment, dataset)
