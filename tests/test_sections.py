"""Check conditional curves against an independent analytic interaction."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.sections import SectionBuilder
from ec_analysis.engine.splits import select_loans
from ec_analysis.schemas.dataset import DatasetLoader


def test_sections_reconstruct_known_interaction(tmp_path: pathlib.Path) -> None:
    random_source = np.random.default_rng(16)
    records = []
    for identifier in range(200):
        for period in ("T0", "T1"):
            first_value, second_value = random_source.uniform(-2, 2, 2)
            records.append(
                dict(
                    id=identifier,
                    period=period,
                    first=first_value,
                    second=second_value,
                    npv=5 + 2 * first_value + 3 * second_value + 4 * first_value * second_value,
                )
            )
    input_path = tmp_path / "input.csv"
    pd.DataFrame(records).to_csv(input_path, index=False)
    dataset = DatasetLoader().load_csv(input_path)
    experiment = BenchmarkRunner(tune_parameters=True).run_experiment(dataset)
    sections = SectionBuilder().build_sections(
        experiment.fitted_models[experiment.analysis_state.selected_model],
        dataset,
        select_loans(dataset, experiment.split_manifest.training_ids, "T0"),
        experiment.analysis_state,
    )
    assert sections.model_name == "quadratic_ridge"
    surface = sections.surfaces[0]
    expected = 4 * np.outer(
        np.array(surface.first_values) - sections.reference["first"],
        np.array(surface.second_values) - sections.reference["second"],
    )
    assert np.asarray(surface.interaction_contrast) == pytest.approx(expected, abs=0.001)
    assert all(abs(value) < 40 for curve in sections.curves for value in curve.centered_predictions)
