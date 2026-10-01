"""Inspect source files and export an explicitly defined paired cohort."""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import typing
import zipfile

import pandas as pd

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.freddie_mac import FreddieMacReader
from ec_analysis.data.quality import CohortSummary, DataQualityReport, FreddieMacProfiler, ProfiledTables


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class FreddieMacPreparation:
    source_reader: FreddieMacReader
    quality_profiler: FreddieMacProfiler

    def inspect_sources(
        self,
        source_paths: tuple[pathlib.Path, ...],
        report_path: pathlib.Path,
        source_kind: typing.Literal["format_sample", "research_sample"] = "format_sample",
    ) -> ProfiledTables | None:
        self._check_output_location(source_paths, report_path)
        source_hashes: dict[str, str] = {}
        try:
            if len(source_paths) not in (1, 2):
                raise ValueError("Supply one archive or two source text files")
            source_hashes = {str(source_path): file_checksum(source_path) for source_path in source_paths}
            if len(source_paths) == 1:
                source_tables = self.source_reader.read_archive(source_paths[0])
            else:
                source_tables = self.source_reader.read_files(source_paths[0], source_paths[1])
            inspected = self.quality_profiler.inspect_tables(source_tables, source_kind, source_hashes)
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            self._save_report(
                DataQualityReport(source_kind=source_kind, source_sha256=source_hashes, errors=[str(error)]),
                report_path,
            )
            return None
        self._save_report(inspected.report, report_path)
        return inspected

    def prepare_cohort(
        self,
        inspected: ProfiledTables,
        initial_month: str,
        final_month: str,
        output_path: pathlib.Path,
        report_path: pathlib.Path,
    ) -> DataQualityReport:
        source_paths = tuple(pathlib.Path(name) for name in inspected.report.source_sha256)
        self._check_output_location(source_paths, output_path)
        self._check_output_location(source_paths, report_path)
        if output_path.resolve() == report_path.resolve():
            raise ValueError("CSV output and quality report must have different paths")
        if not inspected.report.ready_for_pairing:
            raise ValueError("Source quality checks failed; see the saved report")
        try:
            paired_records, cohort_summary = self._build_cohort(inspected, initial_month, final_month)
        except ValueError as error:
            inspected.report.errors.append(str(error))
            inspected.report.ready_for_pairing = False
            self._save_report(inspected.report, report_path)
            raise
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paired_records.to_csv(output_path, index=False)
        inspected.report.cohort = cohort_summary
        inspected.report.prepared_sha256 = file_checksum(output_path)
        self._save_report(inspected.report, report_path)
        return inspected.report

    def _build_cohort(
        self, inspected: ProfiledTables, initial_month: str, final_month: str
    ) -> tuple[pd.DataFrame, CohortSummary]:
        self._validate_periods(initial_month, final_month)
        performance = inspected.normalized.performance
        initial_records = performance.loc[performance["reporting_month"] == initial_month]
        final_records = performance.loc[performance["reporting_month"] == final_month]
        if initial_records.empty or final_records.empty:
            raise ValueError("One or both requested months are absent from performance data")
        initial_ids = set(initial_records["id"])
        final_ids = set(final_records["id"])
        paired_ids = initial_ids & final_ids
        if not paired_ids:
            raise ValueError("Requested months have no common loans")
        if initial_records["current_upb"].isna().any() or final_records["current_upb"].isna().any():
            raise ValueError("Requested months contain missing current UPB; balance totals would be incomplete")
        performance_fields = [
            "id",
            "reporting_month",
            "current_upb",
            "interest_rate",
            "loan_age",
            "remaining_term",
            "estimated_ltv",
            "delinquency_status",
        ]
        original_fields = [
            "id",
            "credit_score",
            "original_ltv",
            "original_dti",
            "original_upb",
            "original_interest_rate",
            "original_term",
            "property_type",
            "occupancy_status",
            "channel",
        ]
        selected_records = performance.loc[
            performance["id"].isin(paired_ids) & performance["reporting_month"].isin([initial_month, final_month]),
            performance_fields,
        ]
        joined_records = selected_records.merge(
            inspected.normalized.origination[original_fields], on="id", how="left", validate="many_to_one"
        )
        joined_records["period"] = joined_records["reporting_month"].map({initial_month: "T0", final_month: "T1"})
        joined_records["as_of_date"] = pd.to_datetime(joined_records["reporting_month"], format="%Y%m").dt.strftime(
            "%Y-%m-%d"
        )
        joined_records = joined_records.drop(columns="reporting_month").sort_values(["id", "period"])
        ordered_columns = ["id", "period", "as_of_date"] + [
            column for column in joined_records if column not in {"id", "period", "as_of_date"}
        ]
        summary = CohortSummary(
            initial_month=initial_month,
            final_month=final_month,
            paired_loans=len(paired_ids),
            initial_only_loans=len(initial_ids - final_ids),
            final_only_loans=len(final_ids - initial_ids),
            output_rows=len(joined_records),
            initial_all_balance=float(initial_records["current_upb"].sum()),
            final_all_balance=float(final_records["current_upb"].sum()),
            initial_retained_balance=float(
                initial_records.loc[initial_records["id"].isin(paired_ids), "current_upb"].sum()
            ),
            final_retained_balance=float(final_records.loc[final_records["id"].isin(paired_ids), "current_upb"].sum()),
        )
        return joined_records[ordered_columns], summary

    def _validate_periods(self, initial_month: str, final_month: str) -> None:
        for month_value in (initial_month, final_month):
            if len(month_value) != 6 or not month_value.isdecimal():
                raise ValueError("Months must use YYYYMM format")
            datetime.datetime.strptime(month_value, "%Y%m")
        if initial_month >= final_month:
            raise ValueError("Initial month must precede final month")

    def _check_output_location(self, source_paths: tuple[pathlib.Path, ...], output_path: pathlib.Path) -> None:
        if output_path.resolve() in {source_path.resolve() for source_path in source_paths}:
            raise ValueError("Output must not overwrite a source file")

    def _save_report(self, report: DataQualityReport, report_path: pathlib.Path) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
