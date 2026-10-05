"""Structural checks before passing numbers to an interpreter."""

import pytest
from pydantic import ValidationError

from ec_analysis.agents.fact_checker import FactChecker
from ec_analysis.schemas.state import AnalysisState


def test_fact_checker_accepts_balanced_state() -> None:
    analysis_state = AnalysisState(
        initial_value=100.0,
        final_value=115.0,
        total_change=15.0,
        selected_model="ridge",
        unexplained_remainder=15.0,
    )

    assert FactChecker().validate_state(analysis_state) == analysis_state


def test_state_rejects_unbalanced_factor_explanation() -> None:
    with pytest.raises(ValidationError, match="do not balance"):
        AnalysisState(
            initial_value=100.0,
            final_value=115.0,
            total_change=15.0,
            selected_model="ridge",
            unexplained_remainder=11.0,
        )
