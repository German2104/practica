"""Download the pinned public UCI source and build an explicitly modelled NPV panel."""

from __future__ import annotations

import argparse
import pathlib
import sys

import httpx

from ec_analysis.data.acquisition import SourceDownloader
from ec_analysis.data.uci_credit import UCI_CREDIT_ASSET, ReceivableValuation, UciPortfolioPreparation


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--source-dir", type=pathlib.Path, default=pathlib.Path("data/raw/uci-credit"))
    argument_parser.add_argument("--output", type=pathlib.Path, required=True)
    argument_parser.add_argument("--initial-month", type=int, default=8)
    argument_parser.add_argument("--final-month", type=int, default=9)
    argument_parser.add_argument("--annual-coupon", type=float, default=0.18)
    argument_parser.add_argument("--annual-discount", type=float, default=0.06)
    argument_parser.add_argument("--lgd", type=float, default=0.60)
    argument_parser.add_argument("--pd-multiplier", type=float, default=1.0)
    arguments = argument_parser.parse_args()
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as http_client:
            SourceDownloader(
                http_client=http_client, source_id="uci-default-credit-card-clients-350", source_kind="observed_panel"
            ).download_assets((UCI_CREDIT_ASSET,), arguments.source_dir)
        preparation = UciPortfolioPreparation(
            valuation=ReceivableValuation(
                annual_coupon=arguments.annual_coupon,
                annual_discount=arguments.annual_discount,
                loss_given_default=arguments.lgd,
                probability_multiplier=arguments.pd_multiplier,
            )
        )
        output_path = preparation.save_dataset(
            arguments.source_dir / UCI_CREDIT_ASSET.filename,
            arguments.output,
            arguments.initial_month,
            arguments.final_month,
        )
        print(output_path)
    except (ValueError, OSError, httpx.HTTPError) as failure:
        print(f"UCI preparation failed: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
