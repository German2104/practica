"""Local non-overlapping model terms before portfolio aggregation."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class LocalTerms:
    values: FloatArray
    factors: tuple[tuple[str, ...], ...]
    base_value: float
    method: str
