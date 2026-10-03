"""Construct fixed model candidates with preprocessing fitted only during training."""

from __future__ import annotations

import dataclasses
import importlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ec_analysis.engine.basis import PairwiseRidgeRegressor
from ec_analysis.engine.stable_gam import CrossFactorRidge, StableGAMPipeline
from ec_analysis.schemas.dataset import PortfolioDataset

ModelPolicy = typing.Literal["benchmark", "fixed_gam"]


class RegressionModel(typing.Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> object: ...
    def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]: ...


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ModelCandidate:
    model_name: str
    estimator: RegressionModel | None
    parameters: dict[str, str | int | float | bool]
    unavailable_reason: str | None = None


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class CandidateFactory:
    model_policy: ModelPolicy = "benchmark"
    random_seed: int = 42
    include_optional: bool = False
    include_extended: bool = False
    parameter_profile: int = 0
    worker_count: int = 2

    def create_candidates(self, dataset: PortfolioDataset) -> list[ModelCandidate]:
        model_names = ["mean", "ridge", "gam_spline"]
        if self.model_policy == "fixed_gam":
            model_names = ["stable_gam"]
        elif self.include_optional:
            model_names.extend(["xgboost", "ebm"])
        if self.include_extended and self.model_policy == "benchmark":
            model_names.extend(["quadratic_ridge", "gam_pairwise"])
        candidates: list[ModelCandidate] = []
        for model_name in model_names:
            try:
                estimator, parameters = self._create_regressor(model_name)
                pipeline = Pipeline(
                    [
                        (
                            "preprocess",
                            self._create_preprocessor(
                                dataset, use_splines=model_name in {"gam_spline", "gam_pairwise", "stable_gam"}
                            ),
                        ),
                        ("regressor", estimator),
                    ]
                )
                if model_name == "stable_gam":
                    pipeline = StableGAMPipeline(
                        steps=pipeline.steps,
                        numeric_features=dataset.numeric_features,
                        categorical_features=dataset.categorical_features,
                    )
                candidates.append(
                    ModelCandidate(
                        model_name=model_name, estimator=typing.cast(RegressionModel, pipeline), parameters=parameters
                    )
                )
            except (ImportError, OSError, ValueError) as failure:
                candidates.append(
                    ModelCandidate(
                        model_name=model_name,
                        estimator=None,
                        parameters={},
                        unavailable_reason=f"{type(failure).__name__}: {failure}",
                    )
                )
        return candidates

    def _create_regressor(self, model_name: str) -> tuple[object, dict[str, str | int | float | bool]]:
        if self.parameter_profile not in {0, 1, 2}:
            raise ValueError("Parameter profile must be 0, 1 or 2")
        if model_name == "mean":
            return DummyRegressor(strategy="mean"), {"strategy": "mean"}
        if model_name in {"ridge", "gam_spline"}:
            alpha_value = (1.0, 0.001, 0.1)[self.parameter_profile]
            parameters: dict[str, str | int | float | bool] = {"alpha": alpha_value}
            if model_name == "gam_spline":
                parameters.update(n_knots=6, degree=3, knots="uniform", extrapolation="linear", interactions=False)
            return Ridge(alpha=alpha_value), parameters
        if model_name in {"quadratic_ridge", "gam_pairwise"}:
            alpha_value = (
                (1.0, 0.001, 0.1)[self.parameter_profile]
                if model_name == "quadratic_ridge"
                else (0.1, 1.0, 10.0)[self.parameter_profile]
            )
            parameters = {"alpha": alpha_value, "basis_degree": 2, "maximum_terms": 2500}
            if model_name == "gam_pairwise":
                parameters.update(n_knots=6, degree=3, knots="uniform", extrapolation="linear")
            return PairwiseRidgeRegressor(alpha=alpha_value), parameters
        if model_name == "stable_gam":
            alpha_value = (0.1, 1.0, 10.0)[self.parameter_profile]
            parameters = {
                "alpha": alpha_value,
                "maximum_terms": 2500,
                "n_knots": 6,
                "degree": 3,
                "knots": "uniform",
                "extrapolation": "linear",
                "cross_factor_only": True,
                "center_terms": True,
                "interaction_penalty": 1000.0,
            }
            return CrossFactorRidge(alpha=alpha_value), parameters
        if model_name == "xgboost":
            parameters = dict(
                n_estimators=(300, 1500, 1200)[self.parameter_profile],
                max_depth=(3, 3, 4)[self.parameter_profile],
                learning_rate=(0.05, 0.03, 0.03)[self.parameter_profile],
                n_jobs=self.worker_count,
                random_state=self.random_seed,
                tree_method="hist",
                objective="reg:squarederror",
            )
            return importlib.import_module("xgboost").XGBRegressor(**parameters), parameters
        parameters = dict(
            interactions=5, max_rounds=2000, outer_bags=4, n_jobs=self.worker_count, random_state=self.random_seed
        )
        if self.parameter_profile:
            parameters.update(
                interactions=10,
                max_rounds=5000 if self.parameter_profile == 2 else 10000,
                max_interaction_bins=32 if self.parameter_profile == 2 else 64,
            )
        return importlib.import_module("interpret.glassbox").ExplainableBoostingRegressor(**parameters), parameters

    def _create_preprocessor(self, dataset: PortfolioDataset, use_splines: bool) -> ColumnTransformer:
        transformers: list[tuple[str, object, list[str]]] = []
        if dataset.numeric_features:
            numeric_steps: list[tuple[str, object]] = [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
            if use_splines:
                numeric_steps.append(
                    (
                        "splines",
                        SplineTransformer(
                            n_knots=6, degree=3, knots="uniform", include_bias=False, extrapolation="linear"
                        ),
                    )
                )
            numeric_steps.append(("scale", StandardScaler()))
            transformers.append(("numeric", Pipeline(numeric_steps), list(dataset.numeric_features)))
        if dataset.categorical_features:
            categorical_steps = [
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("categorical", Pipeline(categorical_steps), list(dataset.categorical_features)))
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
