"""Compare all model families on three observed, nonoverlapping UCI month pairs."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.uci_credit import UCI_CREDIT_ASSET, UciPortfolioPreparation
from ec_analysis.engine.artifacts import ExperimentWriter
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.schemas.dataset import DatasetLoader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=pathlib.Path, default=pathlib.Path("data/raw/uci-credit") / UCI_CREDIT_ASSET.filename
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    parser.add_argument("--model-policy", choices=("benchmark", "fixed_gam"), default="benchmark")
    arguments = parser.parse_args()
    try:
        arguments.output_dir.mkdir(parents=True, exist_ok=False)
        started_at = time.perf_counter()
        summaries = []
        writer = ExperimentWriter(project_root=pathlib.Path(__file__).resolve().parents[1])
        for initial_month, final_month in ((4, 5), (6, 7), (8, 9)):
            window_name = f"2005-{initial_month:02d}-{final_month:02d}"
            print(f"Running UCI window: {window_name}", file=sys.stderr, flush=True)
            input_path = arguments.output_dir / "datasets" / f"{window_name}.csv"
            UciPortfolioPreparation().save_dataset(arguments.source, input_path, initial_month, final_month)
            dataset = DatasetLoader(target_metric="model_npv").load_csv(input_path)
            experiment = BenchmarkRunner(
                model_policy=arguments.model_policy,
                random_seed=arguments.seed,
                include_optional=True,
                tune_parameters=True,
                bootstrap_repeats=arguments.bootstrap_repeats,
                partition_method="stable_hash",
            ).run_experiment(dataset)
            writer.save_experiment(
                experiment, dataset, input_path, file_checksum(input_path), arguments.output_dir / window_name
            )
            summaries.append(
                {
                    "window": window_name,
                    "dataset_sha256": file_checksum(input_path),
                    "manifest_sha256": file_checksum(arguments.output_dir / window_name / "manifest.json"),
                    "analysis": experiment.analysis_state.model_dump(mode="json"),
                }
            )
        result = {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_sha256": file_checksum(arguments.source),
            "protocol_version": "forward_grouped_hash_v2",
            "seed": arguments.seed,
            "model_policy": arguments.model_policy,
            "bootstrap_repeats": arguments.bootstrap_repeats,
            "tuning": True,
            "elapsed_seconds": time.perf_counter() - started_at,
            "artifact_directory": str(arguments.output_dir),
            "windows": summaries,
            "limitation": "Observed panel with assumed model_npv; not default prediction or observed bank NPV.",
        }
        (arguments.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(arguments.output_dir / "summary.json")
    except (ValueError, OSError) as failure:
        print(f"UCI benchmark failed: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
