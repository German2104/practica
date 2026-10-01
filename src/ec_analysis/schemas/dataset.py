"""Validate paired observations and preserve explicit numeric/categorical factors."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np
import pandas as pd

METADATA_COLUMNS: typing.Final = {"id", "period", "as_of_date"}


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PortfolioDataset:
    records: pd.DataFrame
    target_metric: str
    feature_names: tuple[str, ...]

    def period_records(self, period_name: str) -> pd.DataFrame:
        return self.records.loc[self.records["period"] == period_name].sort_values("id")

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(name for name in self.feature_names if pd.api.types.is_numeric_dtype(self.records[name]))

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(name for name in self.feature_names if name not in self.numeric_features)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class DatasetLoader:
    target_metric: str = "npv"
    requested_features: tuple[str, ...] | None = None

    def load_csv(self, input_path: pathlib.Path) -> PortfolioDataset:
        if self.target_metric in METADATA_COLUMNS:
            raise ValueError("Target cannot be an identifier, period or observation date")
        records = pd.read_csv(input_path, dtype={"id": str, "period": str})
        self._validate_records(records)
        feature_names = self._select_features(records)
        for column_name in feature_names:
            if pd.api.types.is_numeric_dtype(records[column_name]):
                numeric_values = records[column_name].dropna().to_numpy(dtype=float)
                if not np.isfinite(numeric_values).all():
                    raise ValueError(f"Factor {column_name} contains infinite values")
            else:
                records[column_name] = records[column_name].astype(object).where(records[column_name].notna(), np.nan)
        return PortfolioDataset(records=records, target_metric=self.target_metric, feature_names=feature_names)

    def _validate_records(self, records: pd.DataFrame) -> None:
        required_columns = {"id", "period", self.target_metric}
        missing_columns = required_columns.difference(records.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if records.empty or records[list(required_columns)].isna().any().any():
            raise ValueError("Dataset is empty or has missing id, period, or target values")
        if records["id"].str.strip().eq("").any():
            raise ValueError("Loan identifiers cannot be blank")
        if set(records["period"]) != {"T0", "T1"}:
            raise ValueError("Dataset must contain exactly T0 and T1 periods")
        if records.duplicated(["id", "period"]).any():
            raise ValueError("Each (id, period) pair must be unique")
        if not records.groupby("id")["period"].nunique().eq(2).all():
            raise ValueError("Every id must appear in both periods")
        if not pd.api.types.is_numeric_dtype(records[self.target_metric]):
            raise ValueError("Target metric must be numeric")
        if not np.isfinite(records[self.target_metric].to_numpy(dtype=float)).all():
            raise ValueError("Target metric must contain finite values")
        if "as_of_date" in records:
            parsed_dates = pd.to_datetime(records["as_of_date"], format="%Y-%m-%d", errors="coerce", utc=True)
            if parsed_dates.isna().any():
                raise ValueError("as_of_date must contain valid YYYY-MM-DD dates")
            if parsed_dates[records["period"] == "T0"].max() >= parsed_dates[records["period"] == "T1"].min():
                raise ValueError("All T0 dates must precede all T1 dates")

    def _select_features(self, records: pd.DataFrame) -> tuple[str, ...]:
        forbidden_columns = METADATA_COLUMNS | {self.target_metric}
        feature_names = self.requested_features or tuple(str(name) for name in records if name not in forbidden_columns)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("Provide a nonempty, unique factor list")
        if set(feature_names) & forbidden_columns:
            raise ValueError("Factors cannot include target, identifiers, periods or dates")
        if not set(feature_names).issubset(records.columns):
            raise ValueError("Requested factors are absent from the dataset")
        return feature_names
