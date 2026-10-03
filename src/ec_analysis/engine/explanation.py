"""Validate candidates before selection, then explain and check the frozen winner."""

from __future__ import annotations

import dataclasses
import json
import typing

import pandas as pd

from ec_analysis.engine.attribution import AttributionOutcome, PipelineAttributor
from ec_analysis.engine.parameter_stability import ParameterStability
from ec_analysis.engine.reliability import QualityPolicy
from ec_analysis.engine.search import SearchOutcome
from ec_analysis.engine.splits import select_loans
from ec_analysis.engine.stability import AttributionBootstrap
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import AnalysisState, BenchmarkResult, SplitManifest

if typing.TYPE_CHECKING:
    from ec_analysis.engine.benchmark import BenchmarkExperiment


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ValidationDiagnostics:
    search_result: SearchOutcome
    attribution_tables: dict[str, pd.DataFrame]
    bootstrap_tables: dict[str, pd.DataFrame]


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ExplanationRunner:
    bootstrap_repeats: int = 0
    random_seed: int = 2026
    attributor: PipelineAttributor = dataclasses.field(default_factory=PipelineAttributor)
    quality_policy: QualityPolicy = dataclasses.field(default_factory=QualityPolicy)

    def evaluate_validation(
        self, search_result: SearchOutcome, dataset: PortfolioDataset, split_manifest: SplitManifest
    ) -> ValidationDiagnostics:
        validation_records = select_loans(dataset, split_manifest.validation_ids)
        training_records = select_loans(dataset, split_manifest.training_ids, "T0")
        calculation_tables: dict[str, pd.DataFrame] = {}
        bootstrap_tables: dict[str, pd.DataFrame] = {}
        checked_candidates: list[BenchmarkResult] = []
        for original_candidate in search_result.benchmark_results:
            candidate = original_candidate.model_copy(deep=True)
            checked_candidates.append(candidate)
            if candidate.status != "completed":
                continue
            estimator = search_result.fitted_models[candidate.model_name]
            attributed: AttributionOutcome | None = None
            try:
                attributed = self.attributor.explain_model(
                    candidate.model_name, estimator, dataset, validation_records, "validation"
                )
                candidate.attribution = attributed.summary
                calculation_tables[attributed.summary.calculation_ref] = attributed.local_calculations
            except (ValueError, RuntimeError, ArithmeticError, ImportError, OSError) as failure:
                candidate.attribution_failure = f"{type(failure).__name__}: {failure}"
            if self.bootstrap_repeats and attributed is not None:
                bootstrap = AttributionBootstrap(
                    requested_repeats=self.bootstrap_repeats, random_seed=self.random_seed, attributor=self.attributor
                )
                stability = bootstrap.evaluate_stability(
                    candidate.model_name, estimator, dataset, training_records, validation_records, attributed
                )
                candidate.stability = stability.summary
                candidate.stability_score = stability.summary.stability_score
                rows = [
                    {"repeat": repeat_index, "factors": json.dumps(factors), "contribution": contribution}
                    for repeat_index, vector in zip(stability.replicate_indices, stability.replicates, strict=True)
                    for factors, contribution in vector.items()
                ]
                bootstrap_tables[f"bootstrap-{candidate.model_name}.csv"] = pd.DataFrame(
                    rows, columns=["repeat", "factors", "contribution"]
                )
            assert candidate.validation is not None
            candidate.reliability = self.quality_policy.assess_explanation(
                candidate.validation, candidate.attribution, candidate.stability, candidate.attribution_failure
            )
            if candidate.tuning_trials and candidate.model_name != "mean" and attributed is not None:
                candidate.parameter_sensitivity = ParameterStability(attributor=self.attributor).compare_variants(
                    candidate.model_name,
                    dataset,
                    validation_records,
                    attributed,
                    search_result.variant_models.get(candidate.model_name, []),
                    candidate.tuning_trials,
                )
                if candidate.parameter_sensitivity.status != "completed":
                    candidate.reliability.status = (
                        "rejected" if candidate.parameter_sensitivity.status == "unstable" else "incomplete"
                    )
                    candidate.reliability.reasons.append(
                        f"Parameter attribution sensitivity: {candidate.parameter_sensitivity.status}."
                    )
        return ValidationDiagnostics(
            search_result=dataclasses.replace(search_result, benchmark_results=checked_candidates),
            attribution_tables=calculation_tables,
            bootstrap_tables=bootstrap_tables,
        )

    def enrich_experiment(self, experiment: BenchmarkExperiment, dataset: PortfolioDataset) -> BenchmarkExperiment:
        calculation_tables = dict(experiment.attribution_tables)
        state_payload = experiment.analysis_state.model_dump()
        limitations = [
            "Attributions describe a fitted model, not causal effects or global feature importance.",
            "Portfolio effects include all input loans; validation and test metrics use held-out loans only.",
            "One T0/T1 transition is not a multi-window temporal reliability assessment.",
            "Prototype quality criteria are configurable research checks, not a bank risk certification.",
        ]
        selected_model = experiment.analysis_state.selected_model
        selected_estimator = experiment.fitted_models[selected_model]
        selected_candidate = next(item for item in experiment.analysis_state.benchmark_summary if item.is_selected)
        selected_outcomes: dict[str, AttributionOutcome] = {}
        test_failure: str | None = None
        for scope, records in (
            ("portfolio", dataset.records),
            ("test", select_loans(dataset, experiment.split_manifest.testing_ids)),
        ):
            try:
                outcome = self.attributor.explain_model(
                    selected_model,
                    selected_estimator,
                    dataset,
                    records,
                    typing.cast(typing.Literal["portfolio", "test"], scope),
                )
                selected_outcomes[scope] = outcome
                calculation_tables[outcome.summary.calculation_ref] = outcome.local_calculations
            except (ValueError, RuntimeError, ArithmeticError, ImportError, OSError) as failure:
                failure_message = f"{type(failure).__name__}: {failure}"
                limitations.append(f"{scope} attribution refused: {failure_message}")
                if scope == "test":
                    test_failure = failure_message
        if "portfolio" in selected_outcomes:
            summary = selected_outcomes["portfolio"].summary
            state_payload.update(
                factor_contributions=[item.model_dump() for item in summary.factor_contributions],
                interactions=[item.model_dump() for item in summary.interactions],
                unexplained_remainder=summary.unexplained_remainder,
                unexplained_fraction=summary.remainder_fraction,
                portfolio_attribution=summary.model_dump(),
                attribution_version="checked_model_delta_v2",
                selection_status="attribution_checked",
            )
        test_attribution = selected_outcomes["test"].summary if "test" in selected_outcomes else None
        if test_attribution is not None:
            state_payload["test_attribution"] = test_attribution.model_dump()
        assert experiment.analysis_state.final_test is not None
        test_reliability = self.quality_policy.assess_explanation(
            experiment.analysis_state.final_test, test_attribution, selected_candidate.stability, test_failure
        )
        state_payload["test_reliability"] = test_reliability.model_dump()
        if selected_candidate.reliability is not None:
            state_payload["validation_reliability"] = selected_candidate.reliability.model_dump()
            if (
                selected_candidate.reliability.status == "accepted"
                and test_reliability.status == "accepted"
                and "portfolio" in selected_outcomes
            ):
                state_payload.update(selection_status="explanation_validated", confidence_level="medium")
        if state_payload["confidence_level"] == "low":
            limitations.append("Reliable explanation was not established; inspect validation and test refusal reasons.")
        if not experiment.split_manifest.calendar_dates_verified:
            limitations.append("No calendar dates; only declared relative periods T0 before T1 are checked.")
        if self.bootstrap_repeats == 0:
            limitations.append("Bootstrap stability was not evaluated.")
        else:
            limitations.append("Bootstrap measures training-sample sensitivity on validation only.")
            if selected_candidate.stability is not None:
                limitations.extend(selected_candidate.stability.warnings)
        state_payload["limitations"] = limitations
        if selected_candidate.parameter_sensitivity is None:
            limitations.append("Parameter attribution sensitivity was not evaluated.")
        elif selected_candidate.parameter_sensitivity.status != "completed":
            limitations.append("Parameter sensitivity did not pass; a reliable explanation is withheld.")
        checked_state = AnalysisState.model_validate(state_payload)
        return dataclasses.replace(experiment, analysis_state=checked_state, attribution_tables=calculation_tables)
