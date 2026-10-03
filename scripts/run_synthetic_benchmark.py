"""Generate known scenarios and persist one identical-protocol experiment per scenario."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.data.synthetic import SCENARIO_NAMES
from ec_analysis.engine.artifacts import ExperimentWriter
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.schemas.dataset import DatasetLoader


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    argument_parser.add_argument("--samples", type=int, default=1500)
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES, default=list(SCENARIO_NAMES))
    argument_parser.add_argument("--advanced", action="store_true")
    argument_parser.add_argument("--tune", action="store_true")
    argument_parser.add_argument("--bootstrap-repeats", type=int, default=0)
    argument_parser.add_argument("--model-policy", choices=("benchmark", "fixed_gam"), default="benchmark")
    arguments = argument_parser.parse_args()
    try:
        if len(set(arguments.scenarios)) != len(arguments.scenarios):
            raise ValueError("Scenarios must be unique")
        if arguments.samples < 10:
            raise ValueError("At least ten loans are required")
        arguments.output_dir.mkdir(parents=True, exist_ok=False)
        started_at = time.perf_counter()
        experiment_writer = ExperimentWriter(project_root=pathlib.Path(__file__).resolve().parents[1])
        summaries: list[dict[str, object]] = []
        for scenario_name in arguments.scenarios:
            print(f"Running scenario: {scenario_name}", file=sys.stderr, flush=True)
            input_path = arguments.output_dir / "datasets" / f"{scenario_name}.csv"
            PortfolioGenerator(
                sample_count=arguments.samples, random_seed=arguments.seed, scenario=scenario_name
            ).generate_csv(input_path)
            dataset = DatasetLoader().load_csv(input_path)
            experiment = BenchmarkRunner(
                model_policy=arguments.model_policy,
                include_optional=arguments.advanced,
                random_seed=arguments.seed,
                bootstrap_repeats=arguments.bootstrap_repeats,
                tune_parameters=arguments.tune,
            ).run_experiment(dataset)
            experiment_writer.save_experiment(
                experiment, dataset, input_path, file_checksum(input_path), arguments.output_dir / scenario_name
            )
            summaries.append(
                {
                    "scenario": scenario_name,
                    "dataset_sha256": file_checksum(input_path),
                    "truth_sha256": file_checksum(input_path.with_suffix(".truth.json")),
                    "manifest_sha256": file_checksum(arguments.output_dir / scenario_name / "manifest.json"),
                    "analysis": experiment.analysis_state.model_dump(mode="json"),
                }
            )
        summary = {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "protocol_version": "forward_grouped_v1",
            "sample_count": arguments.samples,
            "seed": arguments.seed,
            "model_policy": arguments.model_policy,
            "advanced": arguments.advanced,
            "tuning": arguments.tune,
            "bootstrap_repeats": arguments.bootstrap_repeats,
            "elapsed_seconds": time.perf_counter() - started_at,
            "artifact_directory": str(arguments.output_dir),
            "scenarios": summaries,
        }
        (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(arguments.output_dir / "summary.json")
    except (ValueError, OSError) as failure:
        print(f"Synthetic comparison failed: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
