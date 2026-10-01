"""Acquire official format samples, inspect sources, or prepare a two-month cohort."""

from __future__ import annotations

import argparse
import pathlib
import sys
import typing

import httpx

from ec_analysis.data.acquisition import SourceDownloader
from ec_analysis.data.freddie_mac import FreddieMacReader
from ec_analysis.data.preparation import FreddieMacPreparation
from ec_analysis.data.quality import FreddieMacProfiler
from ec_analysis.data.sources import RELEASE47_ASSETS


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    operations = argument_parser.add_subparsers(dest="operation", required=True)
    download_parser = operations.add_parser("download-sample")
    download_parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("data/raw/freddie_mac/release47")
    )
    for operation_name in ("inspect", "prepare"):
        operation_parser = operations.add_parser(operation_name)
        source_group = operation_parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument("--archive", type=pathlib.Path)
        source_group.add_argument("--origination", type=pathlib.Path)
        operation_parser.add_argument("--performance", type=pathlib.Path)
        operation_parser.add_argument(
            "--source-kind", choices=["format_sample", "research_sample"], default="format_sample"
        )
        operation_parser.add_argument("--report", type=pathlib.Path, required=True)
        if operation_name == "prepare":
            operation_parser.add_argument("--initial-month", required=True)
            operation_parser.add_argument("--final-month", required=True)
            operation_parser.add_argument("--output", type=pathlib.Path, required=True)
    return argument_parser


def main() -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    if arguments.operation == "download-sample":
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as http_client:
                manifest = SourceDownloader(http_client=http_client).download_assets(
                    RELEASE47_ASSETS, arguments.output_dir
                )
            print(f"Verified {len(manifest.assets)} assets: {arguments.output_dir / 'manifest.json'}")
        except (httpx.HTTPError, OSError, ValueError) as error:
            print(f"Download failed: {error}", file=sys.stderr)
            return 2
        return 0
    if (arguments.archive is not None and arguments.performance is not None) or (
        arguments.origination is not None and arguments.performance is None
    ):
        argument_parser.error("Use --archive alone or both --origination and --performance")
    source_paths = (arguments.archive,) if arguments.archive else (arguments.origination, arguments.performance)
    source_kind: typing.Literal["format_sample", "research_sample"] = arguments.source_kind
    preparation = FreddieMacPreparation(source_reader=FreddieMacReader(), quality_profiler=FreddieMacProfiler())
    try:
        inspected = preparation.inspect_sources(source_paths, arguments.report, source_kind)
        print(f"Quality report: {arguments.report}")
        if inspected is None:
            return 2
        print(
            f"Common loans: {inspected.report.shared_loan_count}; "
            f"ready for pairing: {inspected.report.ready_for_pairing}"
        )
        if not inspected.report.ready_for_pairing:
            return 2
        if arguments.operation == "prepare":
            preparation.prepare_cohort(
                inspected, arguments.initial_month, arguments.final_month, arguments.output, arguments.report
            )
            print(f"Prepared predictors: {arguments.output}; target metric is not calculated")
    except (ValueError, OSError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
