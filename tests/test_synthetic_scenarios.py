"""Independent checks of known generating effects, kept out of training columns."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from ec_analysis.data.synthetic import SCENARIO_NAMES, ScenarioName, SyntheticFormula
from scripts.generate_dataset import PortfolioGenerator


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_scenario_reproducibility_and_truth_balance(tmp_path: pathlib.Path, scenario: ScenarioName) -> None:
    generator = PortfolioGenerator(sample_count=12, random_seed=17, scenario=scenario)
    first_path = generator.generate_csv(tmp_path / "first.csv")
    second_path = generator.generate_csv(tmp_path / "second.csv")
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.with_suffix(".truth.json").read_bytes() == second_path.with_suffix(".truth.json").read_bytes()
    records = pd.read_csv(first_path)
    metadata = json.loads(first_path.with_suffix(".truth.json").read_text())
    assert "component_deltas" not in records.columns
    actual_change = (
        records.loc[records["period"] == "T1", "npv"].sum() - records.loc[records["period"] == "T0", "npv"].sum()
    )
    assert sum(metadata["component_deltas"].values()) == pytest.approx(actual_change, abs=0.0012)


def test_known_threshold_saturation_u_shape_and_interaction() -> None:
    record: dict[str, str | float | int] = {
        "volume": 500,
        "interest_rate": 0.1,
        "pd": 0.05,
        "lgd": 0.4,
        "ltv": 0.75,
        "period": "T0",
    }
    formula = SyntheticFormula(scenario="mixed")
    assert formula.calculate_terms(record)["ltv_threshold"] == 0
    assert formula.calculate_terms(record | {"ltv": 0.85})["ltv_threshold"] == pytest.approx(-30)
    lower_terms = formula.calculate_terms(record | {"volume": 200})
    middle_terms = formula.calculate_terms(record | {"volume": 400})
    upper_terms = formula.calculate_terms(record | {"volume": 600})
    assert (
        middle_terms["volume_saturation"] - lower_terms["volume_saturation"]
        > upper_terms["volume_saturation"] - middle_terms["volume_saturation"]
        > 0
    )
    assert formula.calculate_terms(record | {"ltv": 0.55})["ltv_u_shape"] == pytest.approx(
        formula.calculate_terms(record)["ltv_u_shape"]
    )
    assert formula.calculate_terms(record | {"ltv": 0.65})["ltv_u_shape"] == 0
    assert formula.calculate_terms(record | {"pd": 0.03})["ltv_pd_interaction"] == 0
    assert formula.calculate_terms(record)["ltv_pd_interaction"] == pytest.approx(-4)
    shifting_formula = SyntheticFormula(scenario="regime_shift")
    assert sum(shifting_formula.calculate_terms(record | {"period": "T1"}).values()) - sum(
        shifting_formula.calculate_terms(record).values()
    ) == pytest.approx(-60)


def test_synthetic_suite_cli_produces_completed_comparison(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from scripts.run_synthetic_benchmark import main

    output_directory = tmp_path / "suite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_synthetic_benchmark.py",
            "--output-dir",
            str(output_directory),
            "--samples",
            "30",
            "--scenarios",
            "linear",
            "threshold",
        ],
    )
    assert main() == 0
    summary = json.loads((output_directory / "summary.json").read_text())
    assert [item["scenario"] for item in summary["scenarios"]] == ["linear", "threshold"]
    assert summary["scenarios"][0]["analysis"]["final_test"]["loan_count"] == 6
    assert (output_directory / "linear" / "manifest.json").exists()
    assert (output_directory / "datasets" / "linear.truth.json").exists()
