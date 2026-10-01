"""Generate paired synthetic data and a separate truth manifest."""

from __future__ import annotations

import argparse
import pathlib

from ec_analysis.data.generator import PortfolioGenerator as PortfolioGenerator
from ec_analysis.data.synthetic import SCENARIO_NAMES


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/portfolio_data.csv"))
    argument_parser.add_argument("--samples", type=int, default=500)
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="credit")
    arguments = argument_parser.parse_args()
    generator = PortfolioGenerator(
        sample_count=arguments.samples, random_seed=arguments.seed, scenario=arguments.scenario
    )
    print(generator.generate_csv(arguments.output))


if __name__ == "__main__":
    main()
