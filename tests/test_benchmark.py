"""Behavioral checks for paired data and the first-stage benchmark."""

from __future__ import annotations

import math
import pathlib

import pandas as pd
import pytest

from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.schemas.dataset import DatasetLoader
from scripts.generate_dataset import PortfolioGenerator


def test_generator_is_reproducible(tmp_path: pathlib.Path) -> None:
    generator = PortfolioGenerator(sample_count=10, random_seed=42)
    first_path = generator.generate_csv(tmp_path / "first.csv")
    second_path = generator.generate_csv(tmp_path / "second.csv")

    assert first_path.read_bytes() == second_path.read_bytes()
    records = pd.read_csv(first_path)
    assert len(records) == 20
    assert set(records["period"]) == {"T0", "T1"}


def test_benchmark_balances_checked_attribution_and_remainder(portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    analysis_state = BenchmarkRunner().run(dataset)

    assert analysis_state.selected_model in {"mean", "ridge", "gam_spline"}
    assert len(analysis_state.benchmark_summary) == 3
    assert all(
        result.validation_rmse is not None and result.validation_rmse >= 0
        for result in analysis_state.benchmark_summary
    )
    assert analysis_state.final_test is not None
    assert analysis_state.protocol_version == "forward_grouped_v1"
    contribution_total = sum(item.contribution for item in analysis_state.factor_contributions)
    contribution_total += sum(item.contribution for item in analysis_state.interactions)
    assert math.isclose(analysis_state.total_change, contribution_total + analysis_state.unexplained_remainder)
    assert analysis_state.factor_contributions
    assert analysis_state.portfolio_attribution is not None


def test_loader_rejects_unpaired_credit(portfolio_csv: pathlib.Path) -> None:
    records = pd.read_csv(portfolio_csv)
    records = records.loc[~((records["id"] == 0) & (records["period"] == "T1"))]
    records.to_csv(portfolio_csv, index=False)

    with pytest.raises(ValueError, match="both periods"):
        DatasetLoader().load_csv(portfolio_csv)
