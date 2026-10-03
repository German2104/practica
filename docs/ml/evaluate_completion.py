"""Audit locally created v3 artifacts and compare every saved family with separate truth.

Only the two fixed project-owned v3 directories are loaded. Never use this helper
for downloaded joblib files. This diagnostic does not change selection or tests.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import pathlib
import typing
import zipfile

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.synthetic import ScenarioName
from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.engine.sections import ModelCurve, ModelSections, ModelSurface
from ec_analysis.engine.shapes import ShapeDetector
from ec_analysis.schemas.state import AnalysisState
from scripts.evaluate_recovery import evaluate_sections

INTERPRETATION_SOURCES: typing.Final = frozenset(
    {
        "src/ec_analysis/agents/evidence.py",
        "src/ec_analysis/agents/fact_checker.py",
        "src/ec_analysis/agents/interpreter.py",
        "src/ec_analysis/agents/prompts.py",
    }
)


class SavedManifest(BaseModel):
    dataset_path: pathlib.Path
    dataset_sha256: str
    source_sha256: dict[str, str]
    artifact_sha256: dict[str, str]
    code_version: str


class SavedSource(BaseModel):
    code_version: str
    source_sha256: dict[str, str]


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class CompletionEvaluator:
    project_root: pathlib.Path

    def evaluate_series(self) -> dict[str, object]:
        originals = list(self.project_root.glob("Постановка*.docx"))
        if len(originals) != 1 or file_checksum(originals[0]) != (
            "63d220b52d41be537bb1b2c1a4aa4ab499647c83a576438b4eb246cf38b20740"
        ):
            raise ValueError("Original assignment document changed")
        audit_records: list[dict[str, object]] = []
        recovery_records: list[dict[str, object]] = []
        code_versions: set[str] = set()
        for series_name in ("completion-v3", "uci-completion-v3"):
            series_directory = self.project_root / "data/experiments" / series_name
            summary_path = series_directory / "summary.json"
            if not summary_path.exists():
                raise ValueError(f"Series has not finished: {series_directory}")
            manifests = sorted(series_directory.glob("*/manifest.json"))
            expected_count = 13 if series_name == "completion-v3" else 3
            if len(manifests) != expected_count:
                raise ValueError("Incomplete series")
            for manifest_path in manifests:
                manifest, state, checked_links = self._audit_experiment(manifest_path)
                code_versions.add(manifest.code_version)
                audit_records.append(
                    dict(
                        experiment=str(manifest_path.parent.relative_to(self.project_root)),
                        artifact_count=len(manifest.artifact_sha256),
                        checked_calculation_links=checked_links,
                        selected_model=state.selected_model,
                        selection_status=state.selection_status,
                        code_version=manifest.code_version,
                        interpretation_source_updates={
                            filename: file_checksum(self.project_root / filename)
                            for filename in sorted(INTERPRETATION_SOURCES)
                            if file_checksum(self.project_root / filename) != manifest.source_sha256[filename]
                        },
                    )
                )
                if series_name == "completion-v3":
                    recovery_records.append(self._evaluate_models(manifest_path.parent, state))
        if len(code_versions) != 1:
            raise ValueError("Series were produced with different source versions")
        return dict(
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            method="v3_checksums_balance_and_all_family_conditional_truth_v1",
            source_sha256=file_checksum(pathlib.Path(__file__)),
            experiment_source_snapshot=dict(
                path="data/reports/completion-v3-source.zip",
                sha256=file_checksum(self.project_root / "data/reports/completion-v3-source.zip"),
                policy=(
                    "All frozen files checked; current analytical sources must match. "
                    "Only four explicitly named fact-selection modules may differ."
                ),
                permitted_interpretation_updates=sorted(INTERPRETATION_SOURCES),
            ),
            used_for_selection=False,
            audits=audit_records,
            recovery=recovery_records,
            limitation=(
                "Conditional T0 truth recovery, not true SHAP values, causal evidence or new test scores. "
                "Correlated grids can be off support. OpenRouter live validation is separate."
            ),
        )

    def _audit_experiment(self, manifest_path: pathlib.Path) -> tuple[SavedManifest, AnalysisState, int]:
        manifest = SavedManifest.model_validate_json(manifest_path.read_text())
        directory = manifest_path.parent
        if file_checksum(self.project_root / manifest.dataset_path) != manifest.dataset_sha256:
            raise ValueError("Dataset checksum changed")
        for filename, expected_checksum in manifest.artifact_sha256.items():
            if pathlib.Path(filename).name != filename:
                raise ValueError("Invalid artifact name")
            if file_checksum(directory / filename) != expected_checksum:
                raise ValueError(f"Artifact changed: {directory / filename}")
        self._audit_sources(manifest)
        if (
            manifest.code_version
            != hashlib.sha256(json.dumps(manifest.source_sha256, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("Invalid code version")
        state = AnalysisState.model_validate_json((directory / "analysis.json").read_text())
        if state.dataset_sha256 != manifest.dataset_sha256:
            raise ValueError("State and manifest describe different data")
        if state.dataset_profile is None or state.global_importance is None or state.shape_diagnostics is None:
            raise ValueError("Required v3 diagnostics are absent")
        selected = next(candidate for candidate in state.benchmark_summary if candidate.is_selected)
        if state.selection_status == "explanation_validated":
            if selected.parameter_sensitivity is None or selected.parameter_sensitivity.status != "completed":
                raise ValueError("Accepted explanation lacks completed parameter sensitivity")
            if state.test_reliability is None or state.test_reliability.status != "accepted":
                raise ValueError("Accepted explanation lacks accepted independent test")
            if state.validation_reliability is None or state.validation_reliability.status != "accepted":
                raise ValueError("Accepted explanation lacks accepted validation")
        references = {item.calculation_ref for item in state.factor_contributions}
        references.update(item.calculation_ref for item in state.interactions)
        attributions = [state.portfolio_attribution, state.test_attribution]
        attributions.extend(candidate.attribution for candidate in state.benchmark_summary)
        for attribution in attributions:
            if attribution is not None:
                references.update(item.calculation_ref for item in attribution.factor_contributions)
                references.update(item.calculation_ref for item in attribution.interactions)
        column_cache: dict[str, set[str]] = {}
        for reference in references:
            filename, separator, column_name = reference.partition("#")
            if not separator or filename not in manifest.artifact_sha256:
                raise ValueError(f"Unresolved calculation: {reference}")
            if filename not in column_cache:
                with (directory / filename).open(newline="") as input_file:
                    column_cache[filename] = set(next(csv.reader(input_file)))
            if column_name not in column_cache[filename]:
                raise ValueError(f"Unresolved term: {reference}")
        for index, importance in enumerate(state.global_importance.factors):
            if importance.calculation_ref != f"global-importance.json#factors/{index}":
                raise ValueError("Invalid importance pointer")
        sections = ModelSections.model_validate_json((directory / "sections.json").read_text())
        for index, shape in enumerate(state.shape_diagnostics.factors):
            if shape.factor_name != sections.curves[index].factor:
                raise ValueError("Shape and curve identify different factors")
            if shape.calculation_ref != f"sections.json#curves/{index}":
                raise ValueError("Invalid curve pointer")
        return manifest, state, len(references) + len(state.global_importance.factors) + len(sections.curves)

    def _audit_sources(self, manifest: SavedManifest) -> None:
        snapshot_path = self.project_root / "data/reports/completion-v3-source.zip"
        with zipfile.ZipFile(snapshot_path) as source_archive:
            snapshot_metadata = SavedSource.model_validate_json(source_archive.read("snapshot-manifest.json"))
            if (
                snapshot_metadata.code_version != manifest.code_version
                or snapshot_metadata.source_sha256 != manifest.source_sha256
            ):
                raise ValueError("Frozen source snapshot does not describe this experiment")
            for filename, expected_checksum in manifest.source_sha256.items():
                source_path = pathlib.Path(filename)
                if source_path.is_absolute() or ".." in source_path.parts:
                    raise ValueError("Invalid source path")
                if hashlib.sha256(source_archive.read(filename)).hexdigest() != expected_checksum:
                    raise ValueError(f"Frozen source changed: {filename}")
                if file_checksum(self.project_root / filename) != expected_checksum:
                    if filename not in INTERPRETATION_SOURCES:
                        raise ValueError(f"Analytical source changed: {filename}")

    def _evaluate_models(self, directory: pathlib.Path, state: AnalysisState) -> dict[str, object]:
        template = ModelSections.model_validate_json((directory / "sections.json").read_text())
        scenario = typing.cast(ScenarioName, directory.name)
        evaluations: list[dict[str, object]] = []
        assert state.shape_diagnostics is not None
        detector = ShapeDetector(minimum_variation=state.shape_diagnostics.minimum_variation)
        for candidate in state.benchmark_summary:
            model_path = directory / f"model-{candidate.model_name}.joblib"
            if not model_path.exists():
                evaluations.append(dict(model=candidate.model_name, status=candidate.status))
                continue
            estimator = typing.cast(RegressionModel, joblib.load(model_path))
            sections = self._project_sections(estimator, template, candidate.model_name)
            measured = evaluate_sections(sections, scenario)
            measured.update(
                model_sha256=file_checksum(model_path),
                model_shapes=detector.detect_sections(sections).model_dump(mode="json"),
                reliability=candidate.reliability.status if candidate.reliability else None,
            )
            evaluations.append(measured)
        return dict(scenario=scenario, selected_model=state.selected_model, models=evaluations)

    def _project_sections(self, estimator: RegressionModel, template: ModelSections, model_name: str) -> ModelSections:
        reference = template.reference
        baseline = float(estimator.predict(pd.DataFrame([reference]))[0])
        curves = []
        for curve in template.curves:
            changed = pd.DataFrame([reference] * len(curve.values))
            changed[curve.factor] = curve.values
            curves.append(
                ModelCurve(
                    factor=curve.factor,
                    values=curve.values,
                    centered_predictions=(estimator.predict(changed) - baseline).tolist(),
                )
            )
        surfaces = []
        for surface in template.surfaces:
            first_name, second_name = surface.factors
            changed = pd.DataFrame([reference] * (len(surface.first_values) * len(surface.second_values)))
            changed[first_name] = np.repeat(surface.first_values, len(surface.second_values))
            changed[second_name] = np.tile(surface.second_values, len(surface.first_values))
            first_frame = changed.copy()
            first_frame[second_name] = reference[second_name]
            second_frame = changed.copy()
            second_frame[first_name] = reference[first_name]
            contrasts = (
                estimator.predict(changed) - estimator.predict(first_frame) - estimator.predict(second_frame) + baseline
            )
            surfaces.append(
                ModelSurface(
                    factors=surface.factors,
                    first_values=surface.first_values,
                    second_values=surface.second_values,
                    interaction_contrast=contrasts.reshape(
                        len(surface.first_values), len(surface.second_values)
                    ).tolist(),
                )
            )
        return ModelSections(
            model_name=model_name,
            reference=reference,
            reference_prediction=baseline,
            curves=curves,
            surfaces=surfaces,
        )


def main() -> int:
    project_root = pathlib.Path(__file__).resolve().parents[2]
    arguments_parser = argparse.ArgumentParser(description=__doc__)
    arguments_parser.add_argument("--output", type=pathlib.Path, default="data/reports/completion-v3-audit.json")
    arguments = arguments_parser.parse_args()
    output_path = project_root / arguments.output
    if output_path.exists():
        raise ValueError("Existing audit is preserved; choose a new version before repeating")
    result = CompletionEvaluator(project_root=project_root).evaluate_series()
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
