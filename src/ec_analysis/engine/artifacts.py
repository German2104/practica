"""Save a complete local experiment with checksums and a final completion manifest."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import typing

import joblib

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.engine.benchmark import BenchmarkExperiment
from ec_analysis.engine.sections import SectionBuilder
from ec_analysis.engine.splits import select_loans
from ec_analysis.schemas.dataset import PortfolioDataset


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ExperimentWriter:
    project_root: pathlib.Path

    def save_experiment(
        self,
        experiment: BenchmarkExperiment,
        dataset: PortfolioDataset,
        input_path: pathlib.Path,
        expected_sha256: str,
        output_directory: pathlib.Path,
    ) -> pathlib.Path:
        if file_checksum(input_path) != expected_sha256:
            raise ValueError("Dataset changed during the experiment; refusing inconsistent provenance")
        output_directory.mkdir(parents=True, exist_ok=False)
        experiment.analysis_state.dataset_sha256 = expected_sha256
        (output_directory / "analysis.json").write_text(experiment.analysis_state.model_dump_json(indent=2) + "\n")
        (output_directory / "split.json").write_text(experiment.split_manifest.model_dump_json(indent=2) + "\n")
        for filename, diagnostic in (
            ("profile.json", experiment.analysis_state.dataset_profile),
            ("global-importance.json", experiment.analysis_state.global_importance),
            ("shapes.json", experiment.analysis_state.shape_diagnostics),
        ):
            if diagnostic is not None:
                (output_directory / filename).write_text(diagnostic.model_dump_json(indent=2) + "\n")
        for model_name, estimator in experiment.fitted_models.items():
            joblib.dump(estimator, output_directory / f"model-{model_name}.joblib")
        for candidate in experiment.analysis_state.benchmark_summary:
            if candidate.parameter_sensitivity is not None:
                (output_directory / candidate.parameter_sensitivity.calculation_ref).write_text(
                    candidate.parameter_sensitivity.model_dump_json(indent=2) + "\n"
                )
        for prediction_name, predictions in experiment.predictions.items():
            predictions.to_csv(output_directory / f"{prediction_name}.csv", index=False)
        for filename, calculations in (experiment.attribution_tables | experiment.bootstrap_tables).items():
            calculations.to_csv(output_directory / filename, index=False)
        sections = SectionBuilder().build_sections(
            experiment.fitted_models[experiment.analysis_state.selected_model],
            dataset,
            select_loans(dataset, experiment.split_manifest.training_ids, "T0"),
            experiment.analysis_state,
        )
        (output_directory / "sections.json").write_text(sections.model_dump_json(indent=2) + "\n")
        package_versions: dict[str, str | None] = {}
        for package_name in ("numpy", "pandas", "scikit-learn", "xgboost", "interpret", "shap", "joblib"):
            try:
                package_versions[package_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package_name] = None
        source_paths = [
            *self.project_root.joinpath("src").rglob("*.py"),
            *self.project_root.joinpath("scripts").glob("*.py"),
        ]
        source_paths.extend(self.project_root / name for name in ("pyproject.toml", "uv.lock"))
        source_hashes = {str(path.relative_to(self.project_root)): file_checksum(path) for path in sorted(source_paths)}
        artifact_hashes = {
            path.name: file_checksum(path) for path in sorted(output_directory.iterdir()) if path.is_file()
        }
        manifest = {
            "status": "complete",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "protocol_version": experiment.split_manifest.protocol_version,
            "dataset_path": str(input_path),
            "dataset_sha256": expected_sha256,
            "target_metric": dataset.target_metric,
            "feature_names": dataset.feature_names,
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "source_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
            "code_version": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "preprocessing": "numeric median (empty=0), optional splines, scale; categorical constant+onehot",
            "test_policy": "selected candidate only; no refit; selection precedes test evaluation",
            "memory_measurement": None,
        }
        # A missing manifest denotes an incomplete run if persistence was interrupted.
        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest_path
