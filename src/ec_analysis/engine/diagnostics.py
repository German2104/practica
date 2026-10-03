"""Enrich the frozen experiment with descriptive data and model diagnostics."""

from __future__ import annotations

import dataclasses
import typing

from ec_analysis.data.profiling import DatasetProfiler
from ec_analysis.engine.importance import PermutationImportance
from ec_analysis.engine.sections import SectionBuilder
from ec_analysis.engine.shapes import ShapeDetector
from ec_analysis.engine.splits import select_loans
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import AnalysisState

if typing.TYPE_CHECKING:
    from ec_analysis.engine.benchmark import BenchmarkExperiment


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class DiagnosticRunner:
    random_seed: int = 2027
    profiler: DatasetProfiler = dataclasses.field(default_factory=DatasetProfiler)
    section_builder: SectionBuilder = dataclasses.field(default_factory=SectionBuilder)
    shape_detector: ShapeDetector = dataclasses.field(default_factory=ShapeDetector)
    relative_shape_floor: float = 0.01

    def enrich_experiment(self, experiment: BenchmarkExperiment, dataset: PortfolioDataset) -> BenchmarkExperiment:
        state_payload = experiment.analysis_state.model_dump()
        estimator = experiment.fitted_models[experiment.analysis_state.selected_model]
        training_records = select_loans(dataset, experiment.split_manifest.training_ids, "T0")
        validation_records = select_loans(dataset, experiment.split_manifest.validation_ids)
        profile = self.profiler.profile_dataset(dataset)
        importance = PermutationImportance(random_seed=self.random_seed).measure_importance(
            estimator,
            dataset,
            validation_records,
        )
        for factor_index, factor_record in enumerate(importance.factors):
            factor_record.calculation_ref = f"global-importance.json#factors/{factor_index}"
        sections = self.section_builder.build_sections(estimator, dataset, training_records, experiment.analysis_state)
        target_scale = float(training_records[dataset.target_metric].std(ddof=0))
        shape_detector = dataclasses.replace(
            self.shape_detector,
            minimum_variation=max(self.shape_detector.minimum_variation, self.relative_shape_floor * target_scale),
        )
        shapes = shape_detector.detect_sections(sections)
        shapes.classification_tolerances["relative_target_scale_floor"] = self.relative_shape_floor
        shapes.warnings.append(
            f"Responses below {100 * self.relative_shape_floor:g}% of training target standard deviation "
            "are classified as unknown; "
            "this configurable materiality floor avoids interpreting numerical noise."
        )
        state_payload.update(
            dataset_profile=profile.model_dump(),
            global_importance=importance.model_dump(),
            shape_diagnostics=shapes.model_dump(),
        )
        return dataclasses.replace(experiment, analysis_state=AnalysisState.model_validate(state_payload))
