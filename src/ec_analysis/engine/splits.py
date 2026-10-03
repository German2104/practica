"""A frozen, reproducible split of loans; training uses only the earlier period."""

from __future__ import annotations

import dataclasses
import hashlib
import typing

import pandas as pd
from sklearn.model_selection import train_test_split

from ec_analysis.schemas.dataset import PortfolioDataset
from ec_analysis.schemas.state import SplitManifest


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ForwardLoanSplit:
    random_seed: int = 42
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    partition_method: typing.Literal["shuffle", "stable_hash"] = "shuffle"

    def create_split(self, dataset: PortfolioDataset) -> SplitManifest:
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("Validation and test fractions must lie between zero and one")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("Validation and test fractions must leave training loans")
        identifiers = sorted(str(value) for value in dataset.records["id"].unique())
        if len(identifiers) < 10:
            raise ValueError("At least ten paired loans are required for train/validation/test")
        if self.partition_method == "stable_hash":
            partitions: list[list[str]] = [[], [], []]
            training_limit = 1 - self.validation_fraction - self.test_fraction
            validation_limit = 1 - self.test_fraction
            for identifier in identifiers:
                digest_value = hashlib.sha256(f"{self.random_seed}:{identifier}".encode()).digest()
                split_value = int.from_bytes(digest_value[:8], "big") / 2**64
                partition_index = 0 if split_value < training_limit else 1 if split_value < validation_limit else 2
                partitions[partition_index].append(identifier)
            training_ids, validation_ids, testing_ids = partitions
        else:
            development_ids, testing_ids = train_test_split(
                identifiers, test_size=self.test_fraction, random_state=self.random_seed
            )
            training_ids, validation_ids = train_test_split(
                development_ids,
                test_size=self.validation_fraction / (1 - self.test_fraction),
                random_state=self.random_seed,
            )
        if min(len(training_ids), len(validation_ids), len(testing_ids)) < 2:
            raise ValueError("Each split requires at least two loans")
        return SplitManifest(
            random_seed=self.random_seed,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            training_ids=sorted(training_ids),
            validation_ids=sorted(validation_ids),
            testing_ids=sorted(testing_ids),
            calendar_dates_verified="as_of_date" in dataset.records,
            partition_method=self.partition_method,
            protocol_version="forward_grouped_hash_v2"
            if self.partition_method == "stable_hash"
            else "forward_grouped_v1",
        )


def select_loans(dataset: PortfolioDataset, identifiers: list[str], period_name: str | None = None) -> pd.DataFrame:
    selected = dataset.records.loc[dataset.records["id"].isin(identifiers)]
    if period_name is not None:
        selected = selected.loc[selected["period"] == period_name]
    return selected.sort_values(["id", "period"])
