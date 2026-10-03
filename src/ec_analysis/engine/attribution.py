"""Explain frozen prediction changes, preserving original factors and numeric checks."""

from __future__ import annotations

import dataclasses
import json
import time
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.engine.ebm_explainer import EbmExplainer
from ec_analysis.engine.ownership import FeatureOwnership
from ec_analysis.engine.shap_explainer import TreeShapExplainer
from ec_analysis.engine.terms import LocalTerms
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import AttributionSummary, FactorContribution, InteractionEffect

FloatArray = npt.NDArray[np.float64]


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class AttributionOutcome:
    summary: AttributionSummary
    local_calculations: pd.DataFrame

    def contribution_vector(self) -> dict[tuple[str, ...], float]:
        contributions: dict[tuple[str, ...], float] = {
            (item.factor_name,): item.contribution for item in self.summary.factor_contributions
        }
        contributions.update({item.factors: item.contribution for item in self.summary.interactions})
        return contributions


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PipelineAttributor:
    feature_ownership: FeatureOwnership = dataclasses.field(default_factory=FeatureOwnership)
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5

    def explain_model(
        self,
        model_name: str,
        estimator: RegressionModel,
        dataset: PortfolioDataset,
        evaluation_records: pd.DataFrame,
        scope: typing.Literal["portfolio", "validation", "test"],
    ) -> AttributionOutcome:
        started_at = time.perf_counter()
        if not isinstance(estimator, Pipeline):
            raise ValueError("Attribution requires a fitted preprocessing/regression pipeline")
        initial_records = evaluation_records.loc[evaluation_records["period"] == "T0"].sort_values("id")
        final_records = evaluation_records.loc[evaluation_records["period"] == "T1"].sort_values("id")
        if initial_records.empty or initial_records["id"].tolist() != final_records["id"].tolist():
            raise ValueError("Attribution requires the same paired loans in T0 and T1")
        if initial_records["id"].duplicated().any():
            raise ValueError("Attribution requires unique paired identifiers")
        ordered_records = pd.concat([initial_records, final_records], ignore_index=True)
        features = ordered_records[list(dataset.feature_names)]
        predictions = np.asarray(estimator.predict(features), dtype=float)
        transformed = np.asarray(estimator.named_steps["preprocess"].transform(features), dtype=float)
        ownership = self.feature_ownership.original_factors(estimator, dataset)
        if len(ownership) != transformed.shape[1]:
            raise ValueError("Transformed features do not match their original factor mapping")
        local_terms = self._model_terms(model_name, estimator.named_steps["regressor"], transformed, ownership)
        grouped_values, grouped_factors = self._group_terms(local_terms)
        reconstructed = grouped_values.sum(axis=1) + local_terms.base_value
        local_errors = predictions - reconstructed
        condition_scale = np.maximum(
            np.abs(predictions), abs(local_terms.base_value) + np.abs(grouped_values).sum(axis=1)
        )
        local_tolerances = self.absolute_tolerance + self.relative_tolerance * condition_scale
        if not np.isfinite(grouped_values).all() or not np.isfinite(predictions).all():
            raise ValueError("Nonfinite attribution or predictions")
        if not np.all(np.abs(local_errors) <= local_tolerances):
            raise ValueError(f"Local attribution fails additivity; max error={np.max(np.abs(local_errors))}")
        loan_count = len(initial_records)
        delta_values = grouped_values[loan_count:].sum(axis=0) - grouped_values[:loan_count].sum(axis=0)
        calculation_ref = f"attribution-{scope}-{model_name}.csv"
        factor_contributions: list[FactorContribution] = []
        interactions: list[InteractionEffect] = []
        for factors, contribution in zip(grouped_factors, delta_values, strict=True):
            column_ref = f"{calculation_ref}#{json.dumps(factors, ensure_ascii=False)}"
            if len(factors) == 1:
                factor_contributions.append(
                    FactorContribution(
                        factor_name=factors[0],
                        contribution=float(contribution),
                        effect_type=(
                            "categorical"
                            if factors[0] in dataset.categorical_features
                            else "linear"
                            if model_name in {"ridge", "mean"}
                            else "nonlinear"
                        ),
                        comment="Change of this model's main term; not a causal effect or global importance.",
                        calculation_ref=column_ref,
                    )
                )
            else:
                interactions.append(
                    InteractionEffect(
                        factors=(factors[0], factors[1]),
                        contribution=float(contribution),
                        description="Non-overlapping pair term, counted once; model dependent.",
                        calculation_ref=column_ref,
                    )
                )
        initial_value = float(initial_records[dataset.target_metric].sum())
        final_value = float(final_records[dataset.target_metric].sum())
        observed_change = final_value - initial_value
        predicted_change = float(predictions[loan_count:].sum() - predictions[:loan_count].sum())
        attributed_change = float(delta_values.sum())
        unexplained = observed_change - attributed_change
        denominator_floor = max(1e-8, 1e-6 * max(abs(initial_value), abs(final_value)))
        summary = AttributionSummary(
            model_name=model_name,
            scope=scope,
            method=local_terms.method,
            loan_count=loan_count,
            factor_contributions=factor_contributions,
            interactions=interactions,
            observed_change=observed_change,
            predicted_change=predicted_change,
            attributed_change=attributed_change,
            prediction_remainder=observed_change - predicted_change,
            unexplained_remainder=unexplained,
            remainder_fraction=unexplained / abs(observed_change) if abs(observed_change) > denominator_floor else None,
            remainder_denominator_floor=denominator_floor,
            numerical_error=predicted_change - attributed_change,
            numerical_tolerance=float(local_tolerances.sum()),
            max_local_error=float(np.max(np.abs(local_errors))),
            elapsed_seconds=time.perf_counter() - started_at,
            calculation_ref=calculation_ref,
        )
        calculations = ordered_records[["id", "period"]].copy()
        calculations["observed"] = ordered_records[dataset.target_metric]
        calculations["prediction"] = predictions
        calculations["base_value"] = local_terms.base_value
        calculations["numerical_error"] = local_errors
        calculations["numerical_tolerance"] = local_tolerances
        for column_index, factors in enumerate(grouped_factors):
            calculations[json.dumps(factors, ensure_ascii=False)] = grouped_values[:, column_index]
        return AttributionOutcome(summary=summary, local_calculations=calculations)

    def _model_terms(
        self, model_name: str, regressor: typing.Any, transformed: FloatArray, ownership: tuple[str, ...]
    ) -> LocalTerms:
        if model_name == "mean":
            return LocalTerms(
                values=np.zeros_like(transformed),
                factors=tuple((name,) for name in ownership),
                base_value=float(np.asarray(regressor.constant_).item()),
                method="constant_delta_v1",
            )
        if model_name in {"ridge", "gam_spline"}:
            return LocalTerms(
                values=transformed * np.asarray(regressor.coef_, dtype=float),
                factors=tuple((name,) for name in ownership),
                base_value=float(regressor.intercept_),
                method="grouped_additive_delta_v1",
            )
        if model_name in {"quadratic_ridge", "gam_pairwise", "stable_gam"}:
            return LocalTerms(
                values=regressor.explain_terms(transformed),
                factors=regressor.factor_groups(ownership),
                base_value=regressor.intercept_,
                method="centered_cross_factor_gam_delta_v1"
                if model_name == "stable_gam"
                else "grouped_pairwise_basis_delta_v1",
            )
        if model_name == "ebm":
            return EbmExplainer(model=regressor).explain_local(transformed, ownership)
        if model_name == "xgboost":
            return TreeShapExplainer(
                model=regressor, absolute_tolerance=self.absolute_tolerance, relative_tolerance=self.relative_tolerance
            ).explain_local(transformed, ownership)
        raise ValueError(f"No attribution method for {model_name}")

    def _group_terms(self, local_terms: LocalTerms) -> tuple[FloatArray, tuple[tuple[str, ...], ...]]:
        grouped_factors = tuple(sorted(set(local_terms.factors), key=lambda factors: (len(factors), factors)))
        grouped_values = np.zeros((len(local_terms.values), len(grouped_factors)))
        for source_index, factors in enumerate(local_terms.factors):
            grouped_values[:, grouped_factors.index(factors)] += local_terms.values[:, source_index]
        return grouped_values, grouped_factors
