"""One process per local web job; no LLM and no network calls."""

from __future__ import annotations

import json
import logging
import pathlib
import sys

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.engine.artifacts import ExperimentWriter
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.schemas.dataset import DatasetLoader


def main() -> int:
    directory = pathlib.Path(sys.argv[1])
    try:
        input_path = directory / "input.csv"
        dataset = DatasetLoader(target_metric=sys.argv[2]).load_csv(input_path)
        experiment = BenchmarkRunner(
            model_policy="fixed_gam",
            tune_parameters=True,
            bootstrap_repeats=20,
            worker_count=1,
            random_seed=2026,
        ).run_experiment(dataset)
        ExperimentWriter(project_root=pathlib.Path(sys.argv[4])).save_experiment(
            experiment, dataset, input_path, file_checksum(input_path), directory / "result"
        )
    except Exception as failure:
        # Process boundary: persist the failure and traceback, never report partial work as completed.
        logging.exception("Analysis job failed")
        (directory / "error.json").write_text(json.dumps({"error": f"{type(failure).__name__}: {failure}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
