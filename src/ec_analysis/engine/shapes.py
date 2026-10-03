"""Conservative numerical classification of a model curve, with measurable evidence."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import numpy.typing as npt

from ec_analysis.engine.sections import ModelCurve, ModelSections
from ec_analysis.schemas.diagnostics import ShapeDiagnostic, ShapeKind, ShapeSummary


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ShapeDetector:
    linear_error_threshold: float = 0.025
    minimum_variation: float = 1e-8

    def detect_sections(self, sections: ModelSections) -> ShapeSummary:
        return ShapeSummary(
            model_name=sections.model_name,
            minimum_variation=self.minimum_variation,
            classification_tolerances={
                "linear_error": self.linear_error_threshold,
                "piecewise_error": 0.03,
                "piecewise_relative_improvement": 0.45,
                "turn_excursion": 0.15,
                "tail_slope_ratio": 0.12,
                "flat_branch_slope": 0.1,
            },
            factors=[self.classify_curve(curve, index) for index, curve in enumerate(sections.curves)],
            warnings=[
                "Shape of a conditional model section at training medians/modes, not a causal dependence.",
                "Classification uses the observed 1–99% training range; it cannot establish behavior outside it.",
                "Thresholds are configurable research heuristics; irregular curves can remain complex/unknown.",
            ],
        )

    def classify_curve(self, curve: ModelCurve, curve_index: int = 0) -> ShapeDiagnostic:
        grid_values = np.asarray(curve.values, dtype=float)
        predictions = np.asarray(curve.centered_predictions, dtype=float)
        if len(grid_values) != len(predictions) or len(grid_values) < 9:
            raise ValueError("Shape classification needs matching curves of at least nine points")
        if not np.isfinite(grid_values).all() or not np.isfinite(predictions).all():
            raise ValueError("Shape inputs must be finite")
        if np.ptp(grid_values) <= 0:
            return self._diagnostic(curve, curve_index, "unknown", 0, 0, "No observed factor variation.")
        if np.any(np.diff(grid_values) <= 0):
            raise ValueError("Shape grid must be strictly increasing")
        variation = float(np.ptp(predictions))
        if variation <= self.minimum_variation:
            return self._diagnostic(curve, curve_index, "unknown", variation, 0, "No material model response.")
        scaled_grid = (grid_values - grid_values[0]) / np.ptp(grid_values)
        scaled_predictions = (predictions - predictions.mean()) / variation
        fitted_line = np.polyval(np.polyfit(scaled_grid, scaled_predictions, 1), scaled_grid)
        linear_error = float(np.sqrt(np.mean((fitted_line - scaled_predictions) ** 2)))
        if linear_error <= self.linear_error_threshold:
            return self._diagnostic(
                curve,
                curve_index,
                "linear",
                variation,
                linear_error,
                "Linear fit error is below the configured normalized tolerance.",
            )
        block_indices = np.linspace(0, len(grid_values) - 1, 9).round().astype(int)
        block_changes = np.diff(scaled_predictions[block_indices])
        change_floor = 0.02
        positive_blocks = block_changes > change_floor
        negative_blocks = block_changes < -change_floor
        minimum_index = int(np.argmin(predictions))
        maximum_index = int(np.argmax(predictions))
        for shape_name, turning_index, left_sign in (
            ("u_shaped", minimum_index, -1),
            ("inverted_u_shaped", maximum_index, 1),
        ):
            if not 0.15 <= scaled_grid[turning_index] <= 0.85:
                continue
            left_changes = np.diff(scaled_predictions[: turning_index + 1])
            right_changes = np.diff(scaled_predictions[turning_index:])
            tolerance = 0.006
            if (
                np.mean(left_sign * left_changes >= -tolerance) >= 0.9
                and np.mean(-left_sign * right_changes >= -tolerance) >= 0.9
                and abs(predictions[turning_index] - predictions[0]) >= 0.15 * variation
                and abs(predictions[-1] - predictions[turning_index]) >= 0.15 * variation
            ):
                diagnostic = self._diagnostic(
                    curve,
                    curve_index,
                    typing.cast(ShapeKind, shape_name),
                    variation,
                    linear_error,
                    "Material interior turning point with opposite monotone branches.",
                )
                diagnostic.turning_point = float(grid_values[turning_index])
                return diagnostic
        monotone_curve = not (positive_blocks.any() and negative_blocks.any())
        initial_slope = abs(float(np.polyfit(scaled_grid[:9], scaled_predictions[:9], 1)[0]))
        final_slope = abs(float(np.polyfit(scaled_grid[-9:], scaled_predictions[-9:], 1)[0]))
        tail_ratio = final_slope / max(initial_slope, 1e-12)
        tail_range = float(np.ptp(scaled_predictions[-9:]))
        if monotone_curve and tail_ratio <= 0.12 and tail_range <= 0.05 and initial_slope >= 0.3:
            diagnostic = self._diagnostic(
                curve,
                curve_index,
                "saturation",
                variation,
                linear_error,
                "Material early slope followed by a near-flat upper-range tail.",
            )
            diagnostic.tail_slope_ratio = tail_ratio
            return diagnostic
        piecewise_error, break_index, first_slope, second_slope = self._best_break(scaled_grid, scaled_predictions)
        nearly_flat_branch = min(abs(first_slope), abs(second_slope)) <= 0.1
        if (monotone_curve or nearly_flat_branch) and piecewise_error <= 0.03 and piecewise_error < 0.45 * linear_error:
            slope_ratio = min(abs(first_slope), abs(second_slope)) / max(abs(first_slope), abs(second_slope), 1e-12)
            shape_name = "threshold" if slope_ratio <= 0.2 and nearly_flat_branch else "regime"
            diagnostic = self._diagnostic(
                curve,
                curve_index,
                typing.cast(ShapeKind, shape_name),
                variation,
                linear_error,
                "Two linear branches fit substantially better than one line.",
            )
            diagnostic.breakpoint = float(grid_values[break_index])
            return diagnostic
        return self._diagnostic(
            curve,
            curve_index,
            "complex",
            variation,
            linear_error,
            "Nonlinear response has no sufficiently supported simple shape under these tolerances.",
        )

    def _best_break(
        self, grid_values: npt.NDArray[np.float64], predictions: npt.NDArray[np.float64]
    ) -> tuple[float, int, float, float]:
        best_result = (float("inf"), 0, 0.0, 0.0)
        for break_index in range(6, len(grid_values) - 6):
            first_line = np.polyfit(grid_values[: break_index + 1], predictions[: break_index + 1], 1)
            second_line = np.polyfit(grid_values[break_index:], predictions[break_index:], 1)
            fitted_values = np.concatenate(
                (
                    np.polyval(first_line, grid_values[:break_index]),
                    np.polyval(second_line, grid_values[break_index:]),
                )
            )
            error_value = float(np.sqrt(np.mean((fitted_values - predictions) ** 2)))
            if error_value < best_result[0]:
                best_result = (error_value, break_index, float(first_line[0]), float(second_line[0]))
        return best_result

    def _diagnostic(
        self,
        curve: ModelCurve,
        curve_index: int,
        shape_name: ShapeKind,
        variation: float,
        linear_error: float,
        explanation: str,
    ) -> ShapeDiagnostic:
        return ShapeDiagnostic(
            factor_name=curve.factor,
            shape=shape_name,
            variation=variation,
            relative_linear_error=linear_error,
            explanation=explanation,
            calculation_ref=f"sections.json#curves/{curve_index}",
        )
