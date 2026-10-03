"""Evaluate saved conditional sections against held-separate synthetic T0 formulas."""

from __future__ import annotations

import argparse
import json
import pathlib
import typing

import numpy as np

from ec_analysis.data.synthetic import ScenarioName, SyntheticFormula
from ec_analysis.engine.sections import ModelSections


def evaluate_sections(sections: ModelSections, scenario: ScenarioName) -> dict[str, object]:
    formula = SyntheticFormula(scenario=scenario)
    reference: dict[str, str | float | int] = dict(sections.reference) | {"period": "T0"}

    def target_value(overrides: dict[str, float]) -> float:
        changed_record = reference.copy()
        changed_record.update(overrides)
        return float(sum(formula.calculate_terms(changed_record).values()))

    baseline = target_value({})
    curve_results = []
    for curve in sections.curves:
        actual_values = np.asarray([target_value({curve.factor: value}) - baseline for value in curve.values])
        predicted = np.asarray(curve.centered_predictions)
        curve_results.append(
            {
                "factor": curve.factor,
                "rmse": float(np.sqrt(np.mean((actual_values - predicted) ** 2))),
                "truth_range": float(np.ptp(actual_values)),
                "true_centered": actual_values.tolist(),
                "predicted_centered": predicted.tolist(),
                "grid": curve.values,
            }
        )
    pair_results = []
    for surface in sections.surfaces:
        first_name, second_name = surface.factors
        actual_surface = np.asarray(
            [
                [
                    target_value({first_name: first, second_name: second})
                    - target_value({first_name: first})
                    - target_value({second_name: second})
                    + baseline
                    for second in surface.second_values
                ]
                for first in surface.first_values
            ]
        )
        predicted_surface = np.asarray(surface.interaction_contrast)
        pair_results.append(
            {
                "factors": surface.factors,
                "rmse": float(np.sqrt(np.mean((actual_surface - predicted_surface) ** 2))),
                "true_rms": float(np.sqrt(np.mean(actual_surface**2))),
                "predicted_rms": float(np.sqrt(np.mean(predicted_surface**2))),
            }
        )
    return {"scenario": scenario, "model": sections.model_name, "curves": curve_results, "pairs": pair_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_directory", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads((arguments.experiment_directory / "summary.json").read_text())
    evaluations = []
    for scenario_record in summary["scenarios"]:
        scenario = typing.cast(ScenarioName, scenario_record["scenario"])
        sections = ModelSections.model_validate_json(
            (arguments.experiment_directory / scenario / "sections.json").read_text()
        )
        evaluations.append(evaluate_sections(sections, scenario))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "method": "conditional_section_truth_comparison_v1",
                "experiments": str(arguments.experiment_directory),
                "limitation": (
                    "T0 conditional shape diagnostic, not true SHAP attribution or final-test prediction quality."
                ),
                "scenarios": evaluations,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
