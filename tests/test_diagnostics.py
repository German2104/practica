"""Independent effects, noise, missing data and parameter changes exercise new diagnostics."""

import copy
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ec_analysis.data.profiling import DatasetProfiler
from ec_analysis.data.synthetic import SyntheticFormula
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.engine.candidates import CandidateFactory
from ec_analysis.engine.importance import PermutationImportance
from ec_analysis.engine.parameter_stability import ParameterStability
from ec_analysis.engine.sections import ModelCurve
from ec_analysis.engine.shapes import ShapeDetector
from ec_analysis.schemas.dataset import DatasetLoader, PortfolioDataset
from ec_analysis.schemas.state import TuningTrial


@pytest.mark.parametrize(
    "shape_name",
    [
        "linear",
        "threshold",
        "saturation",
        "u_shaped",
        "inverted_u_shaped",
        "regime",
        "complex",
        "unknown",
    ],
)
def test_detects_known_shapes_without_access_to_formula(shape_name: str) -> None:
    grid_values = np.linspace(0, 1, 41)
    functions = {
        "linear": 3 * grid_values + 2,
        "threshold": np.maximum(grid_values - 0.6, 0),
        "saturation": 1 - np.exp(-7 * grid_values),
        "u_shaped": (grid_values - 0.5) ** 2,
        "inverted_u_shaped": -((grid_values - 0.5) ** 2),
        "regime": grid_values + 2 * np.maximum(grid_values - 0.5, 0),
        "complex": np.sin(4 * np.pi * grid_values),
        "unknown": np.zeros_like(grid_values),
    }
    diagnostic = ShapeDetector().classify_curve(
        ModelCurve(
            factor="arbitrary_name",
            values=grid_values.tolist(),
            centered_predictions=functions[shape_name].tolist(),
        )
    )
    assert diagnostic.shape == shape_name
    assert diagnostic.calculation_ref == "sections.json#curves/0"
    if shape_name in {"u_shaped", "inverted_u_shaped"}:
        assert diagnostic.turning_point == pytest.approx(0.5)


def test_shapes_reject_invalid_grids_and_do_not_claim_outside_range() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ShapeDetector().classify_curve(
            ModelCurve(factor="factor", values=list(range(10))[::-1], centered_predictions=list(range(10)))
        )
    # A branch of a quadratic cannot establish an unobserved turning point.
    grid_values = np.linspace(0.6, 1, 41)
    diagnostic = ShapeDetector().classify_curve(
        ModelCurve(
            factor="factor", values=grid_values.tolist(), centered_predictions=((grid_values - 0.5) ** 2).tolist()
        )
    )
    assert diagnostic.shape != "u_shaped"


def test_threshold_survives_a_small_opposite_prethreshold_slope() -> None:
    grid_values = np.linspace(0, 1, 41)
    diagnostic = ShapeDetector().classify_curve(
        ModelCurve(
            factor="factor",
            values=grid_values.tolist(),
            centered_predictions=(0.02 * grid_values - np.maximum(grid_values - 0.7, 0)).tolist(),
        )
    )
    assert diagnostic.shape == "threshold"
    assert diagnostic.breakpoint == pytest.approx(0.7, abs=0.025)


def test_materiality_floor_does_not_turn_numerical_noise_into_a_u_shape() -> None:
    grid_values = np.linspace(0, 1, 41)
    detector = ShapeDetector(minimum_variation=0.01)
    diagnostic = detector.classify_curve(
        ModelCurve(
            factor="irrelevant",
            values=grid_values.tolist(),
            centered_predictions=(1e-5 * (grid_values - 0.5) ** 2).tolist(),
        )
    )
    assert diagnostic.shape == "unknown"


def test_importance_distinguishes_prediction_relevance_from_zero_change() -> None:
    random_source = np.random.default_rng(44)
    records = pd.DataFrame({"driver": random_source.normal(size=500), "noise": random_source.normal(size=500)})
    records["value"] = 5 * records.driver
    records["id"] = np.arange(500).astype(str)
    records["period"] = "T1"
    dataset = PortfolioDataset(records=records, target_metric="value", feature_names=("driver", "noise"))
    estimator = LinearRegression().fit(records[["driver", "noise"]].iloc[:300], records.value.iloc[:300])
    diagnostic = PermutationImportance(repeats=5).measure_importance(estimator, dataset, records.iloc[300:])
    assert diagnostic.factors[0].factor_name == "driver"
    assert diagnostic.factors[0].mean_rmse_increase > 5
    assert abs(diagnostic.factors[1].mean_rmse_increase) < 1e-10
    repeated = PermutationImportance(repeats=5).measure_importance(estimator, dataset, records.iloc[300:])
    assert diagnostic == repeated


def test_generic_profile_and_alternate_target(tmp_path: pathlib.Path) -> None:
    random_source = np.random.default_rng(6)
    records = []
    for identifier in range(80):
        factor_value = float(random_source.normal())
        for period_name in ("T0", "T1"):
            records.append(
                dict(
                    id=str(identifier),
                    period=period_name,
                    scale=factor_value,
                    correlated=2 * factor_value,
                    segment="a" if identifier % 2 else "b",
                    value=3 * factor_value + (period_name == "T1"),
                )
            )
    frame = pd.DataFrame(records)
    frame.loc[0, "scale"] = np.nan
    frame.loc[1, "segment"] = "new-category"
    frame.loc[3, "scale"] = 1000
    input_path = tmp_path / "input.csv"
    frame.to_csv(input_path, index=False)
    dataset = DatasetLoader(target_metric="value").load_csv(input_path)
    profile = DatasetProfiler().profile_dataset(dataset)
    assert profile.paired_count == 80
    assert profile.target_distribution is not None
    assert profile.target_distribution.kind == "numeric"
    assert profile.target_distribution.period_statistics["T1"]["mean"] is not None
    assert profile.correlations[0].pearson > 0.99
    assert next(column for column in profile.columns if column.name == "scale").missing_count == 1
    assert (
        next(column for column in profile.columns if column.name == "scale").period_statistics["T1"][
            "iqr_outlier_count"
        ]
        >= 1
    )
    assert next(column for column in profile.columns if column.name == "segment").warnings
    result = BenchmarkRunner().run(dataset)
    assert result.target_metric == "value"
    assert result.dataset_profile is not None and result.global_importance is not None
    assert result.total_change == pytest.approx(
        result.unexplained_remainder
        + sum(item.contribution for item in result.factor_contributions)
        + sum(item.contribution for item in result.interactions)
    )


def test_parameter_check_detects_sign_flip(portfolio_csv: pathlib.Path) -> None:
    dataset = DatasetLoader().load_csv(portfolio_csv)
    training_records = dataset.period_records("T0")
    candidate = next(item for item in CandidateFactory().create_candidates(dataset) if item.model_name == "ridge")
    assert isinstance(candidate.estimator, Pipeline)
    candidate.estimator.fit(training_records[list(dataset.feature_names)], training_records.npv)
    reference = PipelineAttributor().explain_model("ridge", candidate.estimator, dataset, dataset.records, "validation")
    alternative = copy.deepcopy(candidate.estimator)
    alternative.named_steps["regressor"].coef_ *= -1
    trials = [
        TuningTrial(
            profile=number, parameters={}, status="completed", validation_rmse=1, signed_remainder=0, elapsed_seconds=0
        )
        for number in (0, 1)
    ]
    diagnostic = ParameterStability().compare_variants(
        "ridge", dataset, dataset.records, reference, [(0, candidate.estimator), (1, alternative)], trials
    )
    assert diagnostic.status == "unstable"
    assert diagnostic.minimum_sign_agreement == 0
    assert diagnostic.variants[1].contributions


def test_new_formula_has_total_plateau_and_inverted_turn() -> None:
    reference = dict(volume=500, interest_rate=0.1, pd=0.05, lgd=0.4, ltv=0.65, period="T0")
    formula = SyntheticFormula(scenario="full_saturation")
    low_change = sum(formula.calculate_terms(reference | {"volume": 200}).values()) - sum(
        formula.calculate_terms(reference | {"volume": 100}).values()
    )
    tail_change = sum(formula.calculate_terms(reference | {"volume": 1000}).values()) - sum(
        formula.calculate_terms(reference | {"volume": 900}).values()
    )
    assert abs(tail_change) < 0.01 * abs(low_change)
    inverted = SyntheticFormula(scenario="inverted_u_shaped")
    middle = sum(inverted.calculate_terms(reference).values())
    assert middle > sum(inverted.calculate_terms(reference | {"ltv": 0.4}).values())
    assert middle > sum(inverted.calculate_terms(reference | {"ltv": 0.9}).values())
