"""Bounded, fully recorded validation-only parameter search within model families."""

from __future__ import annotations

import dataclasses
import time
import typing
import warnings

import pandas as pd

from ec_analysis.engine.candidates import CandidateFactory, ModelCandidate, ModelPolicy, RegressionModel
from ec_analysis.engine.evaluation import ModelEvaluator
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import BenchmarkResult, TuningTrial


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class SearchOutcome:
    benchmark_results: list[BenchmarkResult]
    fitted_models: dict[str, RegressionModel]
    predictions: dict[str, pd.DataFrame]
    variant_models: dict[str, list[tuple[int, RegressionModel]]] = dataclasses.field(default_factory=dict)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ModelSearch:
    model_policy: ModelPolicy = "benchmark"
    random_seed: int = 42
    include_optional: bool = False
    tune_parameters: bool = False
    worker_count: int = 2
    evaluator: ModelEvaluator = dataclasses.field(default_factory=ModelEvaluator)

    def fit_candidates(
        self, dataset: PortfolioDataset, training_records: pd.DataFrame, validation_records: pd.DataFrame
    ) -> SearchOutcome:
        profiles = (0, 1, 2) if self.tune_parameters or self.model_policy == "fixed_gam" else (0,)
        variants: dict[str, list[tuple[int, ModelCandidate]]] = {}
        for parameter_profile in profiles:
            candidates = CandidateFactory(
                model_policy=self.model_policy,
                random_seed=self.random_seed,
                include_optional=self.include_optional,
                include_extended=self.tune_parameters,
                parameter_profile=parameter_profile,
                worker_count=self.worker_count,
            ).create_candidates(dataset)
            for candidate in candidates:
                if candidate.model_name == "mean" and parameter_profile:
                    continue
                variants.setdefault(candidate.model_name, []).append((parameter_profile, candidate))
        results: list[BenchmarkResult] = []
        fitted_models: dict[str, RegressionModel] = {}
        prediction_tables: dict[str, pd.DataFrame] = {}
        variant_models: dict[str, list[tuple[int, RegressionModel]]] = {}
        for model_name, family_variants in variants.items():
            family_results = []
            trials = []
            for parameter_profile, candidate in family_variants:
                result, predictions = self._train_candidate(candidate, dataset, training_records, validation_records)
                trials.append(
                    TuningTrial(
                        profile=parameter_profile,
                        parameters=result.parameters,
                        status=result.status,
                        validation_rmse=result.validation_rmse,
                        signed_remainder=result.signed_remainder,
                        elapsed_seconds=result.elapsed_seconds,
                        failure_reason=result.failure_reason,
                    )
                )
                family_results.append((result, candidate.estimator, predictions))
                if result.status == "completed" and candidate.estimator is not None:
                    variant_models.setdefault(model_name, []).append((parameter_profile, candidate.estimator))
            successful = [outcome for outcome in family_results if outcome[0].status == "completed"]
            chosen, estimator, predictions = (
                min(successful, key=lambda outcome: typing.cast(float, outcome[0].validation_rmse))
                if successful
                else family_results[0]
            )
            if self.tune_parameters or self.model_policy == "fixed_gam":
                chosen.tuning_trials = trials
                chosen.elapsed_seconds = sum(trial.elapsed_seconds for trial in trials)
            results.append(chosen)
            if chosen.status == "completed" and estimator is not None and predictions is not None:
                fitted_models[model_name] = estimator
                prediction_tables[f"validation-{model_name}"] = predictions
        return SearchOutcome(
            benchmark_results=results,
            fitted_models=fitted_models,
            predictions=prediction_tables,
            variant_models=variant_models,
        )

    def _train_candidate(
        self,
        candidate: ModelCandidate,
        dataset: PortfolioDataset,
        training_records: pd.DataFrame,
        validation_records: pd.DataFrame,
    ) -> tuple[BenchmarkResult, pd.DataFrame | None]:
        if candidate.estimator is None:
            return BenchmarkResult(
                model_name=candidate.model_name,
                status="unavailable",
                elapsed_seconds=0,
                failure_reason=candidate.unavailable_reason,
                parameters=candidate.parameters,
            ), None
        started_at = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                candidate.estimator.fit(
                    training_records[list(dataset.feature_names)], training_records[dataset.target_metric]
                )
                evaluated = self.evaluator.evaluate_model(
                    candidate.estimator, dataset, validation_records, "validation"
                )
            return BenchmarkResult(
                model_name=candidate.model_name,
                validation_rmse=evaluated.metrics.rmse,
                predicted_change=evaluated.metrics.predicted_change,
                signed_remainder=evaluated.metrics.signed_remainder,
                elapsed_seconds=time.perf_counter() - started_at,
                parameters=candidate.parameters,
                validation=evaluated.metrics,
            ), evaluated.predictions
        except (ValueError, RuntimeError, RuntimeWarning, ArithmeticError) as failure:
            return BenchmarkResult(
                model_name=candidate.model_name,
                status="failed",
                elapsed_seconds=time.perf_counter() - started_at,
                parameters=candidate.parameters,
                failure_reason=f"{type(failure).__name__}: {failure}",
            ), None
