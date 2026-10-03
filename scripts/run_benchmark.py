"""Run a reproducible forward comparison with an independent loan holdout."""

from __future__ import annotations

import argparse
import pathlib
import sys

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.engine.artifacts import ExperimentWriter
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import ModelPolicy
from ec_analysis.schemas.dataset import DatasetLoader


def main(*, default_policy: ModelPolicy = "benchmark") -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "csv_path", nargs="?", type=pathlib.Path, default=pathlib.Path("data/portfolio_data.csv")
    )
    argument_parser.add_argument("--target", default="npv")
    argument_parser.add_argument("--features", nargs="+", help="Explicit point-in-time factor allowlist")
    argument_parser.add_argument(
        "--advanced", action="store_true", help="Include XGBoost and EBM; report unavailable models"
    )
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--tune", action="store_true", help="Compare three parameter profiles per family")
    argument_parser.add_argument(
        "--stable-split", action="store_true", help="Keep loan partitions stable across windows"
    )
    argument_parser.add_argument("--no-attribution", action="store_true")
    argument_parser.add_argument("--bootstrap-repeats", type=int, default=20 if default_policy == "fixed_gam" else 0)
    argument_parser.add_argument(
        "--output-dir", type=pathlib.Path, help="New directory for models, predictions and provenance"
    )
    argument_parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    argument_parser.add_argument("--model-policy", choices=("benchmark", "fixed_gam"), default=default_policy)
    arguments = argument_parser.parse_args()
    try:
        if arguments.output_dir is not None and arguments.output_dir.exists():
            raise ValueError("Output directory already exists; use a new experiment directory")
        expected_sha256 = file_checksum(arguments.csv_path)
        dataset = DatasetLoader(
            target_metric=arguments.target,
            requested_features=tuple(arguments.features) if arguments.features else None,
        ).load_csv(arguments.csv_path)
        experiment = BenchmarkRunner(
            model_policy=arguments.model_policy,
            include_optional=arguments.advanced,
            random_seed=arguments.seed,
            include_attribution=not arguments.no_attribution,
            bootstrap_repeats=arguments.bootstrap_repeats,
            tune_parameters=arguments.tune,
            worker_count=arguments.workers,
            partition_method="stable_hash" if arguments.stable_split else "shuffle",
        ).run_experiment(dataset)
        experiment.analysis_state.dataset_sha256 = expected_sha256
        if arguments.output_dir is not None:
            manifest_path = ExperimentWriter(project_root=pathlib.Path(__file__).resolve().parents[1]).save_experiment(
                experiment, dataset, arguments.csv_path, expected_sha256, arguments.output_dir
            )
            print(f"Experiment saved: {manifest_path}", file=sys.stderr)
        print(experiment.analysis_state.model_dump_json(indent=2))
    except (ValueError, OSError) as failure:
        print(f"Benchmark failed: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
