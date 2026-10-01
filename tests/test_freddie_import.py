"""Release 47 mapping and quality failures, using artificial loans only."""

from __future__ import annotations

import pathlib
import sys
import zipfile

import pandas as pd
import pytest

from ec_analysis.data.freddie_mac import FreddieMacReader
from ec_analysis.data.preparation import FreddieMacPreparation
from ec_analysis.data.quality import DataQualityReport, FreddieMacProfiler
from scripts.prepare_dataset import main


def origination_record(loan_identifier: str) -> list[str]:
    field_values = [""] * 31
    for field_position, field_value in {
        0: "740",
        1: "202601",
        3: "205512",
        7: "P",
        9: "35",
        10: "120000",
        11: "80",
        12: "6.5",
        13: "R",
        17: "SF",
        19: loan_identifier,
        21: "360",
    }.items():
        field_values[field_position] = field_value
    return field_values


def performance_record(loan_identifier: str, month: str, balance: str) -> list[str]:
    field_values = [""] * 35
    for field_position, field_value in {
        0: loan_identifier,
        1: month,
        2: balance,
        3: "00",
        4: "1",
        5: "359",
        10: "6.25",
        25: "999",
    }.items():
        field_values[field_position] = field_value
    return field_values


def write_sources(
    directory: pathlib.Path, original_records: list[list[str]], performance_records: list[list[str]]
) -> tuple[pathlib.Path, pathlib.Path]:
    original_path = directory / "orig_2025Q4.txt"
    performance_path = directory / "perf_2025Q4.txt"
    original_path.write_text("\n".join("|".join(row) for row in original_records) + "\n")
    performance_path.write_text("\n".join("|".join(row) for row in performance_records) + "\n")
    return original_path, performance_path


@pytest.fixture
def preparation() -> FreddieMacPreparation:
    return FreddieMacPreparation(source_reader=FreddieMacReader(), quality_profiler=FreddieMacProfiler())


def test_prepared_cohort_preserves_units_dates_and_coverage(
    tmp_path: pathlib.Path, preparation: FreddieMacPreparation
) -> None:
    source_paths = write_sources(
        tmp_path,
        [origination_record(identifier) for identifier in ("A", "B", "C")],
        [
            performance_record("A", "202601", "100"),
            performance_record("B", "202601", "200"),
            performance_record("A", "202602", "90"),
            performance_record("C", "202602", "300"),
        ],
    )
    report_path = tmp_path / "quality.json"
    inspected = preparation.inspect_sources(source_paths, report_path)
    assert inspected is not None
    output_path = tmp_path / "prepared.csv"

    report = preparation.prepare_cohort(inspected, "202601", "202602", output_path, report_path)

    prepared = pd.read_csv(output_path)
    assert prepared["id"].tolist() == ["A", "A"]
    assert prepared["period"].tolist() == ["T0", "T1"]
    assert prepared["as_of_date"].tolist() == ["2026-01-01", "2026-02-01"]
    assert prepared["interest_rate"].tolist() == pytest.approx([0.0625, 0.0625])
    assert prepared["original_ltv"].tolist() == pytest.approx([0.8, 0.8])
    assert prepared["original_dti"].tolist() == pytest.approx([0.35, 0.35])
    assert prepared["estimated_ltv"].isna().all()
    assert not {"npv", "pd", "lgd", "ec"}.intersection(prepared.columns)
    assert report.cohort is not None
    assert report.cohort.paired_loans == 1
    assert report.cohort.initial_only_loans == report.cohort.final_only_loans == 1
    assert report.cohort.initial_all_balance == 300
    assert report.cohort.final_all_balance == 390
    assert report.cohort.initial_retained_balance == 100
    assert report.cohort.final_retained_balance == 90
    assert report.prepared_sha256 is not None


def test_no_shared_keys_produces_report_and_refuses_export(
    tmp_path: pathlib.Path, preparation: FreddieMacPreparation
) -> None:
    source_paths = write_sources(tmp_path, [origination_record("A")], [performance_record("B", "202601", "100")])
    report_path = tmp_path / "quality.json"
    inspected = preparation.inspect_sources(source_paths, report_path)
    assert inspected is not None

    with pytest.raises(ValueError, match="quality checks failed"):
        preparation.prepare_cohort(inspected, "202601", "202602", tmp_path / "prepared.csv", report_path)

    saved_report = DataQualityReport.model_validate_json(report_path.read_text())
    assert saved_report.shared_loan_count == 0
    assert saved_report.unmatched_performance_loans == 1
    assert not saved_report.ready_for_pairing
    assert not (tmp_path / "prepared.csv").exists()


@pytest.mark.parametrize("invalid_value", ["202613", "20261", "", "20260101"])
def test_invalid_reporting_month_is_rejected(
    tmp_path: pathlib.Path, preparation: FreddieMacPreparation, invalid_value: str
) -> None:
    source_paths = write_sources(tmp_path, [origination_record("A")], [performance_record("A", invalid_value, "100")])
    inspected = preparation.inspect_sources(source_paths, tmp_path / "quality.json")

    assert inspected is not None
    assert not inspected.report.ready_for_pairing
    assert any("reporting_month" in error for error in inspected.report.errors)


@pytest.mark.parametrize("invalid_value", ["bad", "inf", "-1"])
def test_invalid_balances_are_not_silently_coerced(
    tmp_path: pathlib.Path, preparation: FreddieMacPreparation, invalid_value: str
) -> None:
    source_paths = write_sources(
        tmp_path, [origination_record("A")], [performance_record("A", "202601", invalid_value)]
    )
    inspected = preparation.inspect_sources(source_paths, tmp_path / "quality.json")

    assert inspected is not None
    assert not inspected.report.ready_for_pairing
    assert any("current_upb" in error for error in inspected.report.errors)


def test_duplicate_keys_block_join(tmp_path: pathlib.Path, preparation: FreddieMacPreparation) -> None:
    source_paths = write_sources(
        tmp_path, [origination_record("A"), origination_record("A")], [performance_record("A", "202601", "100")]
    )
    inspected = preparation.inspect_sources(source_paths, tmp_path / "quality.json")

    assert inspected is not None
    assert not inspected.report.ready_for_pairing
    assert inspected.report.origination is not None
    assert inspected.report.origination.duplicate_key_rows == 2


def test_bad_width_is_recorded_as_format_failure(tmp_path: pathlib.Path, preparation: FreddieMacPreparation) -> None:
    source_paths = write_sources(tmp_path, [origination_record("A")[:-1]], [performance_record("A", "202601", "100")])
    report_path = tmp_path / "quality.json"

    inspected = preparation.inspect_sources(source_paths, report_path)

    assert inspected is None
    assert "expected 31 Release 47 fields" in report_path.read_text()


def test_source_files_cannot_be_overwritten_by_report(
    tmp_path: pathlib.Path, preparation: FreddieMacPreparation
) -> None:
    source_paths = write_sources(tmp_path, [origination_record("A")], [performance_record("A", "202601", "100")])
    original_bytes = source_paths[0].read_bytes()

    with pytest.raises(ValueError, match="overwrite a source"):
        preparation.inspect_sources(source_paths, source_paths[0])

    assert source_paths[0].read_bytes() == original_bytes


def test_zip_reader_never_extracts_member_paths(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../../orig_2025Q4.txt", "|".join(origination_record("A")))
        archive.writestr("sample_perf_2025.txt", "|".join(performance_record("A", "202601", "100")))

    tables = FreddieMacReader().read_archive(archive_path)

    assert tables.origination["id"].tolist() == ["A"]
    assert tables.performance["current_upb"].tolist() == ["100"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source.zip"]


def test_invalid_period_selection_is_recorded(tmp_path: pathlib.Path, preparation: FreddieMacPreparation) -> None:
    source_paths = write_sources(tmp_path, [origination_record("A")], [performance_record("A", "202601", "100")])
    report_path = tmp_path / "quality.json"
    inspected = preparation.inspect_sources(source_paths, report_path)
    assert inspected is not None

    with pytest.raises(ValueError, match="precede"):
        preparation.prepare_cohort(inspected, "202601", "202601", tmp_path / "prepared.csv", report_path)

    assert "precede" in report_path.read_text()
    assert not (tmp_path / "prepared.csv").exists()


def test_prepare_cli_produces_predictors_and_provenance(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_paths = write_sources(
        tmp_path,
        [origination_record("A")],
        [performance_record("A", "202601", "100"), performance_record("A", "202602", "90")],
    )
    output_path = tmp_path / "predictors.csv"
    report_path = tmp_path / "quality.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset.py",
            "prepare",
            "--origination",
            str(source_paths[0]),
            "--performance",
            str(source_paths[1]),
            "--initial-month",
            "202601",
            "--final-month",
            "202602",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    exit_status = main()

    assert exit_status == 0
    assert len(pd.read_csv(output_path)) == 2
    saved_report = DataQualityReport.model_validate_json(report_path.read_text())
    assert saved_report.prepared_sha256 is not None
    assert len(saved_report.source_sha256) == 2
