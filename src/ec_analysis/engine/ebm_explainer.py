"""EBM local terms in target units, with original-factor ownership."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from ec_analysis.engine.terms import FloatArray, LocalTerms


class EbmRegressionModel(typing.Protocol):
    link_: str
    term_features_: list[tuple[int, ...]]
    intercept_: float

    def eval_terms(self, features: FloatArray) -> FloatArray: ...


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EbmExplainer:
    model: EbmRegressionModel

    def explain_local(self, transformed: FloatArray, ownership: tuple[str, ...]) -> LocalTerms:
        if self.model.link_ != "identity":
            raise ValueError("EBM attribution requires identity link in target units")
        factors = tuple(tuple(sorted({ownership[index] for index in term})) for term in self.model.term_features_)
        if any(len(term) not in {1, 2} for term in factors):
            raise ValueError("Only main and pairwise EBM terms are supported")
        return LocalTerms(
            values=np.asarray(self.model.eval_terms(transformed), dtype=float),
            factors=factors,
            base_value=float(np.asarray(self.model.intercept_).item()),
            method="grouped_ebm_terms_v1",
        )
