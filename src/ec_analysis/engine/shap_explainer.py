"""TreeSHAP main/pair terms grouped back to original factors."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ec_analysis.engine.terms import FloatArray, LocalTerms


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class TreeShapExplainer:
    model: object
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5

    def explain_local(self, transformed: FloatArray, ownership: tuple[str, ...]) -> LocalTerms:
        import shap

        if len(ownership) > 128:
            raise ValueError("Tree interaction attribution is limited to 128 transformed features")
        explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent", model_output="raw")
        original_names = sorted(set(ownership))
        term_factors: tuple[tuple[str, ...], ...] = tuple((name,) for name in original_names) + tuple(
            (first_name, second_name)
            for first_index, first_name in enumerate(original_names)
            for second_name in original_names[first_index + 1 :]
        )
        term_positions = {factors: position for position, factors in enumerate(term_factors)}
        term_blocks: list[FloatArray] = []
        # Only one temporary O(batch * transformed_features^2) tensor is retained.
        for feature_batch in np.array_split(transformed, max(1, (len(transformed) + 127) // 128)):
            matrix = np.asarray(explainer.shap_interaction_values(feature_batch), dtype=float)
            if matrix.shape != (len(feature_batch), len(ownership), len(ownership)):
                raise ValueError("Expected single-output TreeSHAP interaction matrices")
            if not np.allclose(
                matrix, matrix.transpose(0, 2, 1), rtol=self.relative_tolerance, atol=self.absolute_tolerance
            ):
                raise ValueError("TreeSHAP interactions are not symmetric")
            grouped_block = np.zeros((len(feature_batch), len(term_factors)))
            for first_index, first_name in enumerate(ownership):
                grouped_block[:, term_positions[(first_name,)]] += matrix[:, first_index, first_index]
                for second_index in range(first_index + 1, len(ownership)):
                    factors = tuple(sorted({first_name, ownership[second_index]}))
                    grouped_block[:, term_positions[factors]] += (
                        matrix[:, first_index, second_index] + matrix[:, second_index, first_index]
                    )
            term_blocks.append(grouped_block)
        return LocalTerms(
            values=np.concatenate(term_blocks),
            factors=term_factors,
            base_value=float(np.asarray(explainer.expected_value).item()),
            method="tree_path_dependent_pairs_v1",
        )
