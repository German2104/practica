"""Descriptive profile of arbitrary paired CSVs; no automatic causal or leakage claims."""

from __future__ import annotations

import dataclasses
import itertools
import typing

from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.diagnostics import ColumnProfile, CorrelatedFactors, DatasetProfile


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class DatasetProfiler:
    correlation_threshold: float = 0.85

    def profile_dataset(self, dataset: PortfolioDataset) -> DatasetProfile:
        columns = [self._profile_column(dataset, column_name) for column_name in dataset.feature_names]
        training_records = dataset.period_records("T0")
        correlations = []
        for first_name, second_name in itertools.combinations(dataset.numeric_features, 2):
            paired_values = training_records[[first_name, second_name]].dropna()
            if len(paired_values) < 3 or paired_values.nunique().min() < 2:
                continue
            pearson_value = float(paired_values.corr(method="pearson").to_numpy(dtype=float)[0, 1])
            spearman_value = float(paired_values.corr(method="spearman").to_numpy(dtype=float)[0, 1])
            if max(abs(pearson_value), abs(spearman_value)) >= self.correlation_threshold:
                correlations.append(
                    CorrelatedFactors(factors=(first_name, second_name), pearson=pearson_value, spearman=spearman_value)
                )
        warnings = [
            "IQR outliers are descriptive flags; no rows are removed automatically.",
            "Feature availability must be confirmed by the data owner; leakage heuristics are not a guarantee.",
        ]
        if correlations:
            warnings.append("Strongly related factors can make separate importance and conditional curves ambiguous.")
        if "as_of_date" not in dataset.records:
            warnings.append("Calendar dates absent; only declared T0/T1 order is verified.")
        target_values = training_records[dataset.target_metric]
        for factor_name in dataset.numeric_features:
            combined = training_records[[factor_name, dataset.target_metric]].dropna()
            if len(combined) >= 3 and combined[factor_name].nunique() > 1 and target_values.nunique() > 1:
                if abs(float(combined.corr().to_numpy(dtype=float)[0, 1])) > 0.999:
                    warnings.append(f"{factor_name}: near-exact target association; check for a target-derived factor.")
        return DatasetProfile(
            row_count=len(dataset.records),
            paired_count=int(dataset.records["id"].nunique()),
            period_counts={str(name): int(count) for name, count in dataset.records["period"].value_counts().items()},
            target_metric=dataset.target_metric,
            columns=columns,
            target_distribution=self._profile_column(dataset, dataset.target_metric),
            correlations=correlations,
            calendar_dates_verified="as_of_date" in dataset.records,
            warnings=warnings,
        )

    def _profile_column(self, dataset: PortfolioDataset, column_name: str) -> ColumnProfile:
        observed_values = dataset.records[column_name]
        numeric_kind = column_name == dataset.target_metric or column_name in dataset.numeric_features
        period_statistics = {}
        warnings = []
        for period_name in ("T0", "T1"):
            selected_values = dataset.period_records(period_name)[column_name].dropna()
            statistics: dict[str, float | int | None] = {"observed_count": len(selected_values)}
            if numeric_kind and len(selected_values):
                lower_quartile = float(selected_values.quantile(0.25))
                upper_quartile = float(selected_values.quantile(0.75))
                quartile_range = upper_quartile - lower_quartile
                outlier_count = int(
                    (
                        (selected_values < lower_quartile - 1.5 * quartile_range)
                        | (selected_values > upper_quartile + 1.5 * quartile_range)
                    ).sum()
                )
                statistics.update(
                    minimum=float(selected_values.min()),
                    maximum=float(selected_values.max()),
                    mean=float(selected_values.mean()),
                    median=float(selected_values.median()),
                    standard_deviation=float(selected_values.std(ddof=0)),
                    quantile_01=float(selected_values.quantile(0.01)),
                    quantile_99=float(selected_values.quantile(0.99)),
                    lower_quartile=lower_quartile,
                    upper_quartile=upper_quartile,
                    iqr_outlier_count=outlier_count,
                )
            period_statistics[period_name] = statistics
        if observed_values.nunique() <= 1:
            warnings.append("Constant or entirely missing factor.")
        if not numeric_kind:
            initial_categories = set(dataset.period_records("T0")[column_name].dropna().astype(str))
            final_categories = set(dataset.period_records("T1")[column_name].dropna().astype(str))
            if final_categories - initial_categories:
                warnings.append("Previously unseen categories in T1.")
            if observed_values.nunique() > len(observed_values) / 2:
                warnings.append("High cardinality; check whether this is an identifier.")
        if any(fragment in column_name.lower() for fragment in ("future", "next_", "target", "prediction", "t1_")):
            warnings.append("Name suggests possible future/target information; confirm availability.")
        return ColumnProfile(
            name=column_name,
            kind="numeric" if numeric_kind else "categorical",
            missing_count=int(observed_values.isna().sum()),
            unique_count=int(observed_values.nunique()),
            period_statistics=period_statistics,
            category_counts={}
            if numeric_kind
            else {str(name): int(count) for name, count in observed_values.value_counts().head(20).items()},
            warnings=warnings,
        )
