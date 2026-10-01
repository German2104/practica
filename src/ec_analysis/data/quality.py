"""Quality checks and normalized predictors before a loan-panel join."""

from __future__ import annotations

import dataclasses
import datetime
import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ec_analysis.data.freddie_mac import FreddieMacTables


class FileProfile(BaseModel):
    row_count: int
    loan_count: int
    duplicate_key_rows: int
    missing_values: dict[str, int]
    invalid_values: dict[str, int]


class CohortSummary(BaseModel):
    initial_month: str
    final_month: str
    paired_loans: int
    initial_only_loans: int
    final_only_loans: int
    output_rows: int
    initial_all_balance: float
    final_all_balance: float
    initial_retained_balance: float
    final_retained_balance: float


class DataQualityReport(BaseModel):
    format_version: str = "freddie-mac-standard-release47"
    importer_version: str = "1"
    source_kind: typing.Literal["format_sample", "research_sample"]
    generated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
    source_sha256: dict[str, str] = Field(default_factory=dict)
    origination: FileProfile | None = None
    performance: FileProfile | None = None
    shared_loan_count: int = 0
    unmatched_performance_loans: int = 0
    period_row_counts: dict[str, int] = Field(default_factory=dict)
    ready_for_pairing: bool = False
    cohort: CohortSummary | None = None
    prepared_sha256: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class NumericField:
    missing_codes: tuple[str, ...] = ()
    divisor: float = 1.0
    minimum: float | None = None
    maximum: float | None = None


ORIGINATION_NUMERIC: typing.Final = {
    "credit_score": NumericField(missing_codes=("9999",), minimum=300, maximum=850),
    "original_upb": NumericField(minimum=0),
    "original_ltv": NumericField(missing_codes=("999",), divisor=100, minimum=1, maximum=998),
    "original_dti": NumericField(missing_codes=("999",), divisor=100, minimum=0, maximum=65),
    "original_interest_rate": NumericField(divisor=100, minimum=0),
    "original_term": NumericField(minimum=1),
}
PERFORMANCE_NUMERIC: typing.Final = {
    "current_upb": NumericField(minimum=0),
    "interest_rate": NumericField(divisor=100, minimum=0),
    "loan_age": NumericField(),
    "remaining_term": NumericField(),
    "estimated_ltv": NumericField(missing_codes=("999",), divisor=100, minimum=1, maximum=998),
}


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ProfiledTables:
    normalized: FreddieMacTables
    report: DataQualityReport


@typing.final
class FreddieMacProfiler:
    def inspect_tables(
        self,
        source_tables: FreddieMacTables,
        source_kind: typing.Literal["format_sample", "research_sample"],
        source_hashes: dict[str, str],
    ) -> ProfiledTables:
        origination, original_invalid = self._normalize_numeric(source_tables.origination, ORIGINATION_NUMERIC)
        performance, performance_invalid = self._normalize_numeric(source_tables.performance, PERFORMANCE_NUMERIC)
        for column_name in ("first_payment_month", "maturity_month"):
            original_invalid[column_name] = self._invalid_months(origination[column_name])
        performance_invalid["reporting_month"] = self._invalid_months(performance["reporting_month"])
        for column_name, missing_code in {"property_type": "99", "occupancy_status": "9", "channel": "9"}.items():
            origination[column_name] = origination[column_name].replace(missing_code, pd.NA)
        original_profile = self._file_profile(origination, ["id"], original_invalid)
        performance_profile = self._file_profile(performance, ["id", "reporting_month"], performance_invalid)
        original_ids = set(origination["id"].dropna())
        performance_ids = set(performance["id"].dropna())
        report = DataQualityReport(
            source_kind=source_kind,
            source_sha256=source_hashes,
            origination=original_profile,
            performance=performance_profile,
            shared_loan_count=len(original_ids & performance_ids),
            unmatched_performance_loans=len(performance_ids - original_ids),
            period_row_counts={
                str(month): int(count)
                for month, count in performance["reporting_month"].value_counts().sort_index().items()
            },
            warnings=["NPV, EC, PD and LGD are not supplied or estimated by this importer."],
        )
        self._record_errors(report)
        report.ready_for_pairing = not report.errors
        if source_kind == "format_sample":
            report.warnings.append("Public format fixture; not a representative annual research sample.")
        return ProfiledTables(
            normalized=FreddieMacTables(origination=origination, performance=performance), report=report
        )

    def _normalize_numeric(
        self, records: pd.DataFrame, field_rules: dict[str, NumericField]
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        normalized = records.copy()
        invalid_counts: dict[str, int] = {}
        for column_name, field_rule in field_rules.items():
            cleaned_values = records[column_name].replace(list(field_rule.missing_codes), pd.NA)
            numeric_values = pd.to_numeric(cleaned_values, errors="coerce").astype("Float64")
            invalid_mask = cleaned_values.notna() & (numeric_values.isna() | ~np.isfinite(numeric_values))
            if field_rule.minimum is not None:
                invalid_mask |= numeric_values.lt(field_rule.minimum).fillna(False)
            if field_rule.maximum is not None:
                invalid_mask |= numeric_values.gt(field_rule.maximum).fillna(False)
            invalid_counts[column_name] = int(invalid_mask.sum())
            normalized[column_name] = numeric_values / field_rule.divisor
        return normalized, invalid_counts

    def _invalid_months(self, values: pd.Series[str]) -> int:
        parsed_dates = pd.to_datetime(values, format="%Y%m", errors="coerce")
        invalid_mask = values.notna() & (~values.str.fullmatch(r"[0-9]{6}", na=False) | parsed_dates.isna())
        return int(invalid_mask.sum())

    def _file_profile(
        self, records: pd.DataFrame, key_columns: list[str], invalid_counts: dict[str, int]
    ) -> FileProfile:
        return FileProfile(
            row_count=len(records),
            loan_count=int(records["id"].nunique()),
            duplicate_key_rows=int(records.duplicated(key_columns, keep=False).sum()),
            missing_values={str(column): int(count) for column, count in records.isna().sum().items()},
            invalid_values={column: count for column, count in invalid_counts.items() if count},
        )

    def _record_errors(self, report: DataQualityReport) -> None:
        for source_name, profile, required_fields in (
            ("origination", report.origination, ("id",)),
            ("performance", report.performance, ("id", "reporting_month")),
        ):
            if profile is None:
                continue
            if profile.duplicate_key_rows:
                report.errors.append(f"{source_name}: duplicate key rows = {profile.duplicate_key_rows}")
            for column_name in required_fields:
                if profile.missing_values[column_name]:
                    report.errors.append(f"{source_name}: missing required {column_name}")
            for column_name, invalid_count in profile.invalid_values.items():
                report.errors.append(f"{source_name}: invalid {column_name} values = {invalid_count}")
        if report.shared_loan_count == 0:
            report.errors.append("Origination and performance have no common loan identifiers; join refused.")
        if report.unmatched_performance_loans:
            report.errors.append(f"Performance loans without origination: {report.unmatched_performance_loans}")
