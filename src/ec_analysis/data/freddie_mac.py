"""Strict Release 47 reader using the July 2026 official field layout."""

from __future__ import annotations

import csv
import dataclasses
import io
import pathlib
import typing
import zipfile

import pandas as pd

ORIGINATION_COLUMNS: typing.Final = (
    "credit_score",
    "first_payment_month",
    "first_time_buyer",
    "maturity_month",
    "msa",
    "mortgage_insurance",
    "unit_count",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "id",
    "loan_purpose",
    "original_term",
    "borrower_count",
    "seller_name",
    "super_conforming",
    "pre_harp_id",
    "special_program",
    "harp_indicator",
    "valuation_method",
    "interest_only",
    "vantage_score",
)

PERFORMANCE_COLUMNS: typing.Final = (
    "id",
    "reporting_month",
    "current_upb",
    "delinquency_status",
    "loan_age",
    "remaining_term",
    "defect_settlement_month",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_month",
    "interest_rate",
    "non_interest_bearing_upb",
    "last_paid_month",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_costs",
    "taxes_insurance",
    "misc_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "rate_step",
    "payment_deferral",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_interest",
    "disaster_delinquency",
    "assistance_plan",
    "period_modification_costs",
    "interest_bearing_upb",
    "mi_cancellation",
    "servicer_name",
    "bankruptcy_cramdown_costs",
)


@typing.final
@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class FreddieMacTables:
    origination: pd.DataFrame
    performance: pd.DataFrame


@typing.final
@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class FreddieMacReader:
    maximum_rows: int = 2_000_000
    maximum_member_bytes: int = 512_000_000

    def read_archive(self, archive_path: pathlib.Path) -> FreddieMacTables:
        with zipfile.ZipFile(archive_path) as archive:
            origination_name = self._find_member(archive, "origination")
            performance_name = self._find_member(archive, "performance")
            with (
                archive.open(origination_name) as binary_stream,
                io.TextIOWrapper(binary_stream, encoding="utf-8-sig") as text_stream,
            ):
                origination = self._read_records(text_stream, ORIGINATION_COLUMNS, origination_name)
            with (
                archive.open(performance_name) as binary_stream,
                io.TextIOWrapper(binary_stream, encoding="utf-8-sig") as text_stream,
            ):
                performance = self._read_records(text_stream, PERFORMANCE_COLUMNS, performance_name)
        return FreddieMacTables(origination=origination, performance=performance)

    def read_files(self, origination_path: pathlib.Path, performance_path: pathlib.Path) -> FreddieMacTables:
        with origination_path.open(encoding="utf-8-sig", newline="") as text_stream:
            origination = self._read_records(text_stream, ORIGINATION_COLUMNS, origination_path.name)
        with performance_path.open(encoding="utf-8-sig", newline="") as text_stream:
            performance = self._read_records(text_stream, PERFORMANCE_COLUMNS, performance_path.name)
        return FreddieMacTables(origination=origination, performance=performance)

    def _find_member(self, archive: zipfile.ZipFile, file_kind: str) -> str:
        prefixes = (
            ("orig_", "sample_orig_", "origination_sample_")
            if file_kind == "origination"
            else ("perf_", "sample_perf_", "performance_sample_")
        )
        matching_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and pathlib.PurePosixPath(member.filename).name.startswith(prefixes)
            and member.filename.endswith(".txt")
        ]
        if len(matching_members) != 1:
            raise ValueError(f"Archive must contain exactly one {file_kind} text file")
        member = matching_members[0]
        if member.file_size > self.maximum_member_bytes:
            raise ValueError(f"Archive member exceeds size limit: {member.filename}")
        return member.filename

    def _read_records(self, text_stream: typing.TextIO, columns: tuple[str, ...], source_name: str) -> pd.DataFrame:
        parsed_rows: list[list[str]] = []
        for row_number, field_values in enumerate(
            csv.reader(text_stream, delimiter="|", quoting=csv.QUOTE_NONE), start=1
        ):
            if row_number > self.maximum_rows:
                raise ValueError(f"Row limit exceeded in {source_name}")
            if len(field_values) != len(columns):
                raise ValueError(
                    f"{source_name}, row {row_number}: expected {len(columns)} Release 47 fields, "
                    f"received {len(field_values)}"
                )
            parsed_rows.append([field_value.strip() for field_value in field_values])
        if not parsed_rows:
            raise ValueError(f"Empty source file: {source_name}")
        return pd.DataFrame(parsed_rows, columns=list(columns), dtype="string").replace("", pd.NA)
