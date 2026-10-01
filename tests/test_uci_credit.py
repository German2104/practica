"""Temporal mapping, source exclusions and independent valuation controls."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ec_analysis.data.uci_credit import ReceivableValuation, UciPortfolioPreparation


def monthly_records() -> pd.DataFrame:
    records = pd.DataFrame({"ID": range(1, 21)})
    for column_index in range(1, 7):
        records[f"BILL_AMT{column_index}"] = np.arange(20) + 100 * column_index
        records["PAY_0" if column_index == 1 else f"PAY_{column_index}"] = column_index % 4
    records["default payment next month"] = 0
    return records


def test_valuation_matches_cashflow_controls_and_monotonicity() -> None:
    valuation = ReceivableValuation()
    volume_values = pd.Series([100.0, 100.0, 0.0])
    probabilities = pd.Series([0.0, 1.0, 0.40])
    values = valuation.calculate_npv(volume_values, probabilities)
    assert values.tolist() == pytest.approx([101.5 / 1.005 - 100, 40 / 1.005 - 100, 0])
    assert (
        dataclasses.replace(valuation, annual_discount=0.12).calculate_npv(volume_values, probabilities) <= values
    ).all()
    assert (
        dataclasses.replace(valuation, loss_given_default=0.8).calculate_npv(volume_values, probabilities) <= values
    ).all()
    payment_statuses = pd.Series([-2, -1, 0, 1, 2, 3, 4, 9], dtype=float)
    assert valuation.default_probabilities(payment_statuses).tolist() == pytest.approx(
        [0.01, 0.01, 0.01, 0.04, 0.10, 0.20, 0.40, 0.40]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        valuation.calculate_npv(pd.Series([-1.0]), pd.Series([0.1]))


def test_month_mapping_and_future_labels_do_not_enter_training_data() -> None:
    source_records = monthly_records()
    preparation = UciPortfolioPreparation()
    original, metadata = preparation.prepare_periods(source_records, 7, 8)
    changed_records = source_records.copy()
    changed_records["default payment next month"] = 1
    changed_records["BILL_AMT1"] = 1_000_000
    changed_records["PAY_0"] = 8
    changed, _ = preparation.prepare_periods(changed_records, 7, 8)
    pd.testing.assert_frame_equal(original, changed)
    assert set(original.columns) == {"id", "period", "as_of_date", "volume", "pd", "model_npv"}
    assert original.loc[original["period"] == "T0", "as_of_date"].unique().tolist() == ["2005-07-31"]
    assert original.loc[original["period"] == "T1", "as_of_date"].unique().tolist() == ["2005-08-31"]
    assert original.loc[(original["id"] == "1") & (original["period"] == "T0"), "volume"].item() == 300
    assert original.loc[(original["id"] == "1") & (original["period"] == "T1"), "volume"].item() == 200
    assert metadata["feature_names"] == ["volume", "pd"]


def test_negative_balances_are_excluded_as_pairs_and_counted() -> None:
    source_records = monthly_records()
    source_records.loc[0, "BILL_AMT2"] = -1
    source_records.loc[1, "BILL_AMT1"] = -5
    prepared, metadata = UciPortfolioPreparation().prepare_periods(source_records, 8, 9)
    assert len(prepared) == 36
    assert metadata["excluded_accounts"] == 2
    assert set(prepared["id"]).isdisjoint({"1", "2"})
    assert prepared.groupby("id")["period"].nunique().eq(2).all()


@pytest.mark.parametrize("defect", ["duplicate", "missing", "status", "infinite"])
def test_invalid_monthly_records_are_rejected(defect: str) -> None:
    source_records = monthly_records()
    if defect == "duplicate":
        source_records.loc[0, "ID"] = 2
    elif defect == "missing":
        source_records = source_records.drop(columns="PAY_6")
    elif defect == "status":
        source_records.loc[0, "PAY_6"] = 20
    else:
        source_records["BILL_AMT6"] = float("inf")
    with pytest.raises(ValueError):
        UciPortfolioPreparation().prepare_periods(source_records, 8, 9)


def test_invalid_month_order_and_assumptions_are_rejected() -> None:
    with pytest.raises(ValueError, match="months"):
        UciPortfolioPreparation().prepare_periods(monthly_records(), 9, 8)
    with pytest.raises(ValueError, match="finite"):
        ReceivableValuation(annual_discount=float("nan"))
    with pytest.raises(ValueError, match="loss"):
        ReceivableValuation(loss_given_default=2.0)
