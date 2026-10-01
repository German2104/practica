"""Create a reproducible two-period synthetic credit portfolio."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import pathlib
import random
import typing

from ec_analysis.data.synthetic import ScenarioName, SyntheticFormula


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PortfolioGenerator:
    sample_count: int = 500
    random_seed: int = 42
    scenario: ScenarioName = "credit"

    def generate_csv(self, output_path: pathlib.Path) -> pathlib.Path:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        random_source: typing.Final = random.Random(self.random_seed)
        formula = SyntheticFormula(scenario=self.scenario)
        component_totals: dict[str, dict[str, float]] = {"T0": {}, "T1": {}}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output_stream:
            csv_writer = csv.DictWriter(
                output_stream,
                fieldnames=["id", "period", "volume", "interest_rate", "pd", "lgd", "ltv", "npv"],
            )
            csv_writer.writeheader()
            for loan_number in range(self.sample_count):
                initial_record = self._create_initial_record(random_source, loan_number)
                final_record = self._create_final_record(random_source, initial_record)
                csv_writer.writerow(initial_record)
                csv_writer.writerow(final_record)
                for period_record in (initial_record, final_record):
                    totals = component_totals[str(period_record["period"])]
                    for term_name, term_value in formula.calculate_terms(period_record).items():
                        totals[term_name] = totals.get(term_name, 0.0) + term_value
        truth_metadata = {
            "generator_version": "4",
            "scenario": self.scenario,
            "sample_count": self.sample_count,
            "random_seed": self.random_seed,
            "csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "formula_source": "src/ec_analysis/data/synthetic.py",
            "formula_sha256": hashlib.sha256(pathlib.Path(__file__).with_name("synthetic.py").read_bytes()).hexdigest(),
            "component_totals": component_totals,
            "component_deltas": {
                name: component_totals["T1"][name] - value for name, value in component_totals["T0"].items()
            },
            "note": "Generating terms, not SHAP truth; CSV targets are rounded to four decimals.",
        }
        output_path.with_suffix(".truth.json").write_text(json.dumps(truth_metadata, indent=2) + "\n")
        return output_path

    def _create_initial_record(self, random_source: random.Random, loan_number: int) -> dict[str, str | float | int]:
        record: dict[str, str | float | int] = {
            "id": loan_number,
            "period": "T0",
            "volume": round(random_source.uniform(100, 1000), 2),
            "interest_rate": round(random_source.uniform(0.08, 0.15), 6),
            "pd": round(random_source.uniform(0.01, 0.08), 6),
            "lgd": round(random_source.uniform(0.25, 0.65), 6),
            "ltv": round(random_source.uniform(0.4, 0.9), 6),
        }
        if self.scenario == "correlated":
            record["pd"] = round(0.01 + 0.14 * (float(record["ltv"]) - 0.4), 6)
        record["npv"] = round(self._synthetic_npv(record), 4)
        return record

    def _create_final_record(
        self,
        random_source: random.Random,
        initial_record: dict[str, str | float | int],
    ) -> dict[str, str | float | int]:
        final_record: dict[str, str | float | int] = {
            "id": initial_record["id"],
            "period": "T1",
            "volume": round(float(initial_record["volume"]) * random_source.uniform(0.95, 1.1), 2),
            "interest_rate": round(
                max(0.01, float(initial_record["interest_rate"]) - random_source.uniform(0, 0.015)), 6
            ),
            "pd": round(min(1.0, float(initial_record["pd"]) + random_source.uniform(0, 0.012)), 6),
            "lgd": round(min(1.0, float(initial_record["lgd"]) + random_source.uniform(-0.03, 0.03)), 6),
            "ltv": round(min(1.0, float(initial_record["ltv"]) + random_source.uniform(0.01, 0.08)), 6),
        }
        if self.scenario == "correlated":
            final_record["pd"] = round(0.01 + 0.14 * (float(final_record["ltv"]) - 0.4), 6)
        final_record["npv"] = round(self._synthetic_npv(final_record), 4)
        return final_record

    def _synthetic_npv(self, record: dict[str, str | float | int]) -> float:
        return sum(SyntheticFormula(scenario=self.scenario).calculate_terms(record).values())
