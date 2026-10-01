"""Known generating components; these are not model or SHAP attributions."""

from __future__ import annotations

import dataclasses
import math
import typing

ScenarioName = typing.Literal[
    "credit",
    "linear",
    "threshold",
    "saturation",
    "u_shaped",
    "interaction",
    "mixed",
    "regime_shift",
    "correlated",
    "no_signal",
    "inverted_u_shaped",
    "full_saturation",
    "observable_regimes",
]
SCENARIO_NAMES: typing.Final = typing.get_args(ScenarioName)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class SyntheticFormula:
    scenario: ScenarioName = "credit"

    def calculate_terms(self, record: dict[str, str | float | int]) -> dict[str, float]:
        volume_value = float(record["volume"])
        interest_rate = float(record["interest_rate"])
        default_probability = float(record["pd"])
        loss_given_default = float(record["lgd"])
        loan_to_value = float(record["ltv"])
        if self.scenario == "no_signal":
            return {"constant": 100.0}
        if self.scenario == "full_saturation":
            return {"base": 20.0, "full_plateau": 120 * (1 - math.exp(-volume_value / 120))}
        if self.scenario == "observable_regimes":
            return {
                "base": 20.0,
                "first_regime": 0.08 * volume_value,
                "second_regime": 0.4 * max(volume_value - 550, 0),
            }
        if self.scenario == "inverted_u_shaped":
            return {
                "base": 100.0,
                "inverted_u_shape": -400 * (loan_to_value - 0.65) ** 2,
                "rate_effect": 100 * interest_rate,
            }
        if self.scenario == "credit":
            return {
                "base": volume_value * (0.1 + 2 * interest_rate - 3 * default_probability * loss_given_default),
                "ltv_threshold": -volume_value * 0.8 * max(loan_to_value - 0.8, 0),
                "ltv_pd_interaction": -volume_value
                * 8
                * max(loan_to_value - 0.75, 0)
                * max(default_probability - 0.04, 0),
            }
        result_terms = {
            "linear": 0.12 * volume_value
            + 500 * interest_rate
            - 600 * default_probability
            - 20 * loss_given_default
            + 10 * loan_to_value
        }
        if self.scenario in {"threshold", "mixed", "regime_shift", "correlated"}:
            result_terms["ltv_threshold"] = -300 * max(loan_to_value - 0.75, 0)
        if self.scenario in {"saturation", "mixed", "regime_shift", "correlated"}:
            result_terms["volume_saturation"] = 80 * (1 - math.exp(-volume_value / 250))
        if self.scenario in {"u_shaped", "mixed", "regime_shift", "correlated"}:
            result_terms["ltv_u_shape"] = 250 * (loan_to_value - 0.65) ** 2
        if self.scenario in {"interaction", "mixed", "regime_shift", "correlated"}:
            result_terms["ltv_pd_interaction"] = (
                -2000 * max(loan_to_value - 0.65, 0) * max(default_probability - 0.03, 0)
            )
        if self.scenario == "regime_shift":
            result_terms["unobserved_regime_shift"] = -60.0 if record["period"] == "T1" else 0.0
        return result_terms
