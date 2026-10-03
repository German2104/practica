"""Global prediction importance on held-out T1; never part of the change balance."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import pandas as pd

from ec_analysis.engine.candidates import RegressionModel
from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.diagnostics import FactorImportance, GlobalImportance


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PermutationImportance:
    repeats: int = 10
    random_seed: int = 2027
    maximum_samples: int = 2000

    def measure_importance(
        self, estimator: RegressionModel, dataset: PortfolioDataset, validation_records: pd.DataFrame
    ) -> GlobalImportance:
        if self.repeats < 2 or self.maximum_samples < 2:
            raise ValueError("Importance needs at least two repeats and samples")
        evaluation_records = validation_records.loc[validation_records["period"] == "T1"]
        if len(evaluation_records) > self.maximum_samples:
            evaluation_records = evaluation_records.sample(self.maximum_samples, random_state=self.random_seed)
        feature_values = evaluation_records[list(dataset.feature_names)].copy()
        target_values = evaluation_records[dataset.target_metric].to_numpy(dtype=float)
        baseline_error = float(np.sqrt(np.mean((target_values - estimator.predict(feature_values)) ** 2)))
        random_source = np.random.default_rng(self.random_seed)
        factors: list[FactorImportance] = []
        for factor_name in dataset.feature_names:
            repeat_values = []
            for _repeat_index in range(self.repeats):
                permuted_values = feature_values.copy()
                permuted_values[factor_name] = feature_values[factor_name].to_numpy()[
                    random_source.permutation(len(feature_values))
                ]
                error_value = float(np.sqrt(np.mean((target_values - estimator.predict(permuted_values)) ** 2)))
                repeat_values.append(error_value - baseline_error)
            factors.append(
                FactorImportance(
                    factor_name=factor_name,
                    mean_rmse_increase=float(np.mean(repeat_values)),
                    standard_deviation=float(np.std(repeat_values, ddof=1)),
                    minimum=min(repeat_values),
                    maximum=max(repeat_values),
                    repeat_values=repeat_values,
                    calculation_ref=f"global-importance.json#factors/{len(factors)}",
                )
            )
        return GlobalImportance(
            sample_count=len(feature_values),
            baseline_rmse=baseline_error,
            random_seed=self.random_seed,
            repeats=self.repeats,
            factors=sorted(factors, key=lambda item: (-item.mean_rmse_increase, item.factor_name)),
            warnings=[
                "Increase in held-out RMSE, in target units; not a signed portfolio contribution or causal effect.",
                "Correlated factors can share importance; permutation can create rare combinations.",
                "Repeat dispersion measures permutation variability, not a calibrated confidence interval.",
                "Validation used for selection; this diagnostic is not an additional independent test.",
            ],
        )
