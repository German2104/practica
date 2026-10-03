"""Training-loan bootstrap of attribution on a fixed validation portfolio."""

from __future__ import annotations

import dataclasses
import time
import typing

import numpy as np
import pandas as pd
from sklearn.base import clone

from ec_analysis.engine.attribution import AttributionOutcome, PipelineAttributor
from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import StabilitySummary, TermStability


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class BootstrapOutcome:
    summary: StabilitySummary
    replicates: list[dict[tuple[str, ...], float]]
    replicate_indices: list[int]


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class AttributionBootstrap:
    requested_repeats: int = 20
    random_seed: int = 2026
    material_fraction: float = 0.01
    warning_threshold: float = 0.70
    reference_threshold: float = 0.85
    attributor: PipelineAttributor = dataclasses.field(default_factory=PipelineAttributor)

    def evaluate_stability(
        self,
        model_name: str,
        estimator: RegressionModel,
        dataset: PortfolioDataset,
        training_records: pd.DataFrame,
        validation_records: pd.DataFrame,
        reference: AttributionOutcome,
    ) -> BootstrapOutcome:
        if self.requested_repeats < 5:
            raise ValueError("At least five bootstrap repeats are needed for a preliminary diagnostic")
        if not 0 < self.material_fraction < 1 or not 0 <= self.warning_threshold <= self.reference_threshold <= 1:
            raise ValueError("Invalid bootstrap materiality or stability thresholds")
        if set(training_records["period"]) != {"T0"} or training_records["id"].duplicated().any():
            raise ValueError("Bootstrap training must contain one T0 observation per loan")
        if set(training_records["id"]) & set(validation_records["id"]):
            raise ValueError("Bootstrap validation loans must be independent of training")
        if reference.summary.scope != "validation":
            raise ValueError("Bootstrap reference must describe the validation portfolio")
        started_at = time.perf_counter()
        random_source = np.random.default_rng(self.random_seed)
        reference_vector = reference.contribution_vector()
        replicate_vectors: list[dict[tuple[str, ...], float]] = []
        failures: list[str] = []
        replicate_indices: list[int] = []
        if sum(abs(value) for value in reference_vector.values()) > 1e-8:
            for replicate_index in range(self.requested_repeats):
                sampled_indices = random_source.integers(0, len(training_records), size=len(training_records))
                sampled_records = training_records.iloc[sampled_indices]
                try:
                    fitted_model = typing.cast(RegressionModel, clone(estimator))
                    fitted_model.fit(
                        sampled_records[list(dataset.feature_names)], sampled_records[dataset.target_metric]
                    )
                    attributed = self.attributor.explain_model(
                        model_name, fitted_model, dataset, validation_records, "validation"
                    )
                    replicate_vectors.append(attributed.contribution_vector())
                    replicate_indices.append(replicate_index)
                except (ValueError, RuntimeError, ArithmeticError, ImportError, OSError) as failure:
                    failures.append(f"replicate {replicate_index}: {type(failure).__name__}: {failure}")
        summary = self._summarize(reference_vector, replicate_vectors, failures, time.perf_counter() - started_at)
        return BootstrapOutcome(summary=summary, replicates=replicate_vectors, replicate_indices=replicate_indices)

    def _summarize(
        self,
        reference: dict[tuple[str, ...], float],
        replicates: list[dict[tuple[str, ...], float]],
        failures: list[str],
        elapsed_seconds: float,
    ) -> StabilitySummary:
        reference_size = sum(abs(value) for value in reference.values())
        status: typing.Literal["completed", "incomplete", "no_signal"] = (
            "no_signal"
            if reference_size <= 1e-8
            else "completed"
            if len(replicates) == self.requested_repeats
            else "incomplete"
        )
        warnings = ["Exploratory training bootstrap; percentiles are not calibrated confidence intervals."]
        summary = StabilitySummary(
            requested_repeats=self.requested_repeats,
            completed_repeats=len(replicates),
            random_seed=self.random_seed,
            status=status,
            material_fraction=self.material_fraction,
            warning_threshold=self.warning_threshold,
            reference_threshold=self.reference_threshold,
            failures=failures,
            warnings=warnings,
            elapsed_seconds=elapsed_seconds,
        )
        if not replicates:
            summary.warnings.append("No nonzero reference effects or no successful repeats; stability is undefined.")
            return summary
        all_factors = sorted(set(reference).union(*(set(replicate) for replicate in replicates)))
        reference_values = np.asarray([reference.get(factors, 0) for factors in all_factors], dtype=float)
        replicate_values = np.asarray(
            [[replicate.get(factors, 0) for factors in all_factors] for replicate in replicates], dtype=float
        )
        material_floor = self.material_fraction * reference_size
        material_mask = np.abs(reference_values) >= material_floor
        disagreements = np.abs(replicate_values - reference_values).sum(axis=1)
        denominators = np.abs(replicate_values).sum(axis=1) + reference_size
        similarities = np.clip(1 - disagreements / denominators, 0, 1)
        reference_signs = np.where(np.abs(reference_values) >= material_floor, np.sign(reference_values), 0)
        replicate_signs = np.where(np.abs(replicate_values) >= material_floor, np.sign(replicate_values), 0)
        sign_frequencies = (replicate_signs == reference_signs).mean(axis=0)
        rank_correlations: list[float] = []
        reference_ranks = pd.Series(np.abs(reference_values[material_mask])).rank().to_numpy()
        if len(reference_ranks) >= 2 and np.std(reference_ranks) > 0:
            for replicate in replicate_values:
                replicate_ranks = pd.Series(np.abs(replicate[material_mask])).rank().to_numpy()
                if np.std(replicate_ranks) > 0:
                    rank_correlations.append(float(np.corrcoef(reference_ranks, replicate_ranks)[0, 1]))
        for column_index, factors in enumerate(all_factors):
            column_values = replicate_values[:, column_index]
            summary.term_statistics.append(
                TermStability(
                    factors=factors,
                    reference_contribution=float(reference_values[column_index]),
                    mean_contribution=float(np.mean(column_values)),
                    standard_deviation=float(np.std(column_values, ddof=1)) if len(column_values) > 1 else 0.0,
                    percentile_025=float(np.quantile(column_values, 0.025)),
                    percentile_975=float(np.quantile(column_values, 0.975)),
                    sign_agreement=float(sign_frequencies[column_index]),
                    is_material=bool(material_mask[column_index]),
                )
            )
        if status == "completed":
            summary.stability_score = float(np.mean(similarities))
            summary.material_sign_agreement = (
                float(sign_frequencies[material_mask].mean()) if material_mask.any() else None
            )
            summary.mean_rank_correlation = (
                float(np.clip(np.mean(rank_correlations), -1, 1)) if rank_correlations else None
            )
            if summary.stability_score < self.warning_threshold:
                summary.warnings.append("Unstable attribution: score is below the configured warning threshold.")
            elif summary.stability_score < self.reference_threshold:
                summary.warnings.append("Attribution is below the configured reference stability threshold.")
        else:
            summary.warnings.append("Some repeats failed; aggregate stability score is withheld.")
        return summary
