"""Audit the project-owned fixed GAM series and measure a frozen linear baseline."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import typing
import zipfile

import joblib
import numpy as np
import pandas as pd

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.candidates import CandidateFactory, RegressionModel
from ec_analysis.engine.evaluation import ModelEvaluator
from ec_analysis.engine.splits import select_loans
from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.state import AnalysisState, SplitManifest


@typing.final
class SingleGAMAuditor:
    def __init__(self, project_root: pathlib.Path) -> None:
        self.project_root = project_root

    def evaluate_series(self) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for name in ("single-gam-v4", "uci-single-gam-v4"):
            directory = self.project_root / "data/experiments" / name
            summary = json.loads((directory / "summary.json").read_text())
            manifests = sorted(directory.glob("*/manifest.json"))
            if len(manifests) != (13 if name == "single-gam-v4" else 3):
                raise ValueError("Incomplete series")
            if summary["model_policy"] != "fixed_gam" or summary["seed"] != 2028:
                raise ValueError("Unexpected protocol")
            records.extend(self._evaluate_experiment(path) for path in manifests)
        return dict(
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            method="single_gam_checksums_balance_reload_and_linear_baseline_v1",
            source_sha256=file_checksum(pathlib.Path(__file__)),
            frozen_source="data/reports/single-gam-v4-source.zip",
            source_sha256_archive=file_checksum(self.project_root / "data/reports/single-gam-v4-source.zip"),
            audits=records,
            used_for_selection=False,
        )

    def _evaluate_experiment(self, manifest_path: pathlib.Path) -> dict[str, object]:
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        version = hashlib.sha256(json.dumps(manifest["source_sha256"], sort_keys=True).encode()).hexdigest()
        if manifest["code_version"] != version:
            raise ValueError("Invalid code version")
        with zipfile.ZipFile(self.project_root / "data/reports/single-gam-v4-source.zip") as archive:
            frozen = json.loads(archive.read("snapshot-manifest.json"))
            if frozen["source_sha256"] != manifest["source_sha256"] or frozen["code_version"] != version:
                raise ValueError("Snapshot does not describe experiment")
            for filename, expected in manifest["source_sha256"].items():
                if file_checksum(self.project_root / filename) != expected:
                    raise ValueError(f"Current source changed: {filename}")
                if hashlib.sha256(archive.read(filename)).hexdigest() != expected:
                    raise ValueError(f"Frozen source changed: {filename}")
        for filename, expected in manifest["artifact_sha256"].items():
            if pathlib.Path(filename).name != filename or file_checksum(directory / filename) != expected:
                raise ValueError(f"Artifact changed: {filename}")
        input_path = self.project_root / manifest["dataset_path"]
        if file_checksum(input_path) != manifest["dataset_sha256"]:
            raise ValueError("Input changed")
        state = AnalysisState.model_validate_json((directory / "analysis.json").read_text())
        if state.selected_model != "stable_gam" or len(state.benchmark_summary) != 1:
            raise ValueError("Fixed model policy violated")
        dataset = DatasetLoader(target_metric=state.target_metric).load_csv(input_path)
        split = SplitManifest.model_validate_json((directory / "split.json").read_text())
        training = select_loans(dataset, split.training_ids, "T0")
        testing = select_loans(dataset, split.testing_ids)
        selected = state.benchmark_summary[0]
        assert selected.attribution is not None and state.portfolio_attribution is not None
        assert state.test_attribution is not None and state.final_test is not None
        checked_links = 0
        for attribution in (selected.attribution, state.portfolio_attribution, state.test_attribution):
            references = [term.calculation_ref for term in attribution.factor_contributions]
            references.extend(term.calculation_ref for term in attribution.interactions)
            for reference in references:
                filename, separator, column = reference.partition("#")
                if not separator or filename not in manifest["artifact_sha256"]:
                    raise ValueError("Invalid calculation reference")
                with (directory / filename).open(newline="") as source:
                    if column not in next(csv.reader(source)):
                        raise ValueError("Calculation column absent")
                checked_links += 1
        estimator = typing.cast(RegressionModel, joblib.load(directory / "model-stable_gam.joblib"))
        evaluated = ModelEvaluator().evaluate_model(estimator, dataset, testing, "test")
        if evaluated.metrics.model_dump() != state.final_test.model_dump():
            raise ValueError("Reloaded model changed test predictions")
        saved_predictions = pd.read_csv(directory / "test-stable_gam.csv")
        if not np.allclose(saved_predictions.prediction, evaluated.predictions.prediction, rtol=1e-12, atol=1e-9):
            raise ValueError("Saved prediction CSV differs")
        ridge = next(
            candidate for candidate in CandidateFactory().create_candidates(dataset) if candidate.model_name == "ridge"
        )
        assert ridge.estimator is not None
        ridge.estimator.fit(training[list(dataset.feature_names)], training[dataset.target_metric])
        baseline = ModelEvaluator().evaluate_model(ridge.estimator, dataset, testing, "test")
        ridge_attribution = PipelineAttributor().explain_model(
            "ridge", ridge.estimator, dataset, dataset.records, "portfolio"
        )
        return dict(
            experiment=str(directory.relative_to(self.project_root)),
            code_version=version,
            artifact_count=len(manifest["artifact_sha256"]),
            checked_term_links=checked_links,
            selection_status=state.selection_status,
            test=state.final_test.model_dump(),
            stability=selected.stability_score,
            completed_bootstraps=selected.stability.completed_repeats if selected.stability is not None else 0,
            parameter_status=selected.parameter_sensitivity.status if selected.parameter_sensitivity else None,
            baseline=dict(
                model="ridge",
                alpha=1.0,
                test=baseline.metrics.model_dump(),
                portfolio_remainder=ridge_attribution.summary.unexplained_remainder,
                policy="Fixed alpha; same saved training and test groups. No model selection.",
            ),
        )


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default="data/reports/single-gam-v4-audit.json")
    output = root / parser.parse_args().output
    if output.exists():
        raise ValueError("Existing audit is preserved")
    result = SingleGAMAuditor(root).evaluate_series()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
