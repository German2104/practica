"""Shared deterministic test portfolio."""

from __future__ import annotations

import pathlib

import pytest

from scripts.generate_dataset import PortfolioGenerator


@pytest.fixture
def portfolio_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    return PortfolioGenerator(sample_count=40, random_seed=7).generate_csv(tmp_path / "portfolio.csv")
