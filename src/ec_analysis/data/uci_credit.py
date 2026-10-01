"""Prepare observed monthly UCI card balances with an explicitly assumed valuation."""

from __future__ import annotations

import calendar
import dataclasses
import io
import json
import pathlib
import typing
import zipfile

import numpy as np
import pandas as pd

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.sources import SourceAsset

UCI_CREDIT_ASSET: typing.Final = SourceAsset(
    filename="default-credit-clients.zip",
    source_url="https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip",
    expected_sha256="56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602",
)


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ReceivableValuation:
    """One-month expected cash flow less funding; assumptions, not estimated risk."""

    annual_coupon: float = 0.18
    annual_discount: float = 0.06
    loss_given_default: float = 0.60
    probability_multiplier: float = 1.0

    def __post_init__(self) -> None:
        parameters = dataclasses.astuple(self)
        if not all(np.isfinite(value) for value in parameters):
            raise ValueError("Valuation parameters must be finite")
        if not 0 <= self.annual_coupon <= 1 or not 0 <= self.annual_discount <= 1:
            raise ValueError("Annual rates must lie between zero and one")
        if not 0 <= self.loss_given_default <= 1 or self.probability_multiplier <= 0:
            raise ValueError("Invalid loss or probability assumptions")

    def default_probabilities(self, payment_status: pd.Series[float]) -> pd.Series[float]:
        probabilities = np.select(
            [payment_status <= 0, payment_status == 1, payment_status == 2, payment_status == 3],
            [0.01, 0.04, 0.10, 0.20],
            default=0.40,
        )
        return pd.Series(np.minimum(probabilities * self.probability_multiplier, 1.0), index=payment_status.index)

    def calculate_npv(self, volume_values: pd.Series[float], probabilities: pd.Series[float]) -> pd.Series[float]:
        if (
            not np.isfinite(volume_values).all()
            or not np.isfinite(probabilities).all()
            or (volume_values < 0).any()
            or not probabilities.between(0, 1).all()
        ):
            raise ValueError("Valuation requires nonnegative balances and probabilities in [0, 1]")
        expected_cashflow = volume_values * (
            (1 - probabilities) * (1 + self.annual_coupon / 12) + probabilities * (1 - self.loss_given_default)
        )
        return expected_cashflow / (1 + self.annual_discount / 12) - volume_values


@typing.final
class UciCreditReader:
    def read_archive(self, archive_path: pathlib.Path) -> pd.DataFrame:
        if file_checksum(archive_path) != UCI_CREDIT_ASSET.expected_sha256:
            raise ValueError("UCI source checksum mismatch; review the source before importing")
        with zipfile.ZipFile(archive_path) as archive:
            member_name = "default of credit card clients.xls"
            if archive.getinfo(member_name).file_size > 10_000_000:
                raise ValueError("UCI workbook exceeds the import limit")
            workbook_data = archive.read(member_name)
        records = pd.read_excel(io.BytesIO(workbook_data), engine="xlrd", header=1)
        if len(records) != 30_000:
            raise ValueError("Unexpected number of clients in the pinned UCI source")
        self.validate_records(records)
        return records

    def validate_records(self, records: pd.DataFrame) -> None:
        required_columns = ["ID", *[f"BILL_AMT{index}" for index in range(1, 7)]]
        required_columns += ["PAY_0", *[f"PAY_{index}" for index in range(2, 7)]]
        if not set(required_columns).issubset(records.columns):
            raise ValueError("UCI monthly columns are missing")
        if records.empty or records["ID"].isna().any() or records["ID"].duplicated().any():
            raise ValueError("UCI client identifiers must be nonempty and unique")
        numeric_records = records[required_columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric_records.to_numpy(dtype=float)).all():
            raise ValueError("UCI monthly observations must be finite")
        if not np.equal(numeric_records["ID"], np.floor(numeric_records["ID"])).all():
            raise ValueError("UCI identifiers must be integer values")
        status_columns = [name for name in required_columns if name.startswith("PAY_")]
        status_values = numeric_records[status_columns].to_numpy(dtype=float)
        if not ((status_values >= -2) & (status_values <= 9) & (status_values == np.floor(status_values))).all():
            raise ValueError("Unknown UCI payment status")


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class UciPortfolioPreparation:
    valuation: ReceivableValuation = dataclasses.field(default_factory=ReceivableValuation)

    def prepare_periods(
        self, source_records: pd.DataFrame, initial_month: int, final_month: int
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        if not 4 <= initial_month < final_month <= 9:
            raise ValueError("UCI months must satisfy April <= T0 < T1 <= September 2005")
        UciCreditReader().validate_records(source_records)
        monthly_records = [self._monthly_records(source_records, month) for month in (initial_month, final_month)]
        eligible_ids = set(monthly_records[0].loc[monthly_records[0]["volume"] >= 0, "id"])
        eligible_ids &= set(monthly_records[1].loc[monthly_records[1]["volume"] >= 0, "id"])
        if len(eligible_ids) < 10:
            raise ValueError("Too few nonnegative paired accounts")
        output_parts = []
        period_summaries = []
        for period_name, monthly_frame in zip(("T0", "T1"), monthly_records, strict=True):
            selected = monthly_frame.loc[monthly_frame["id"].isin(eligible_ids)].copy()
            selected["period"] = period_name
            selected["model_npv"] = self.valuation.calculate_npv(selected["volume"], selected["pd"])
            period_summaries.append(
                {
                    "period": period_name,
                    "as_of_date": str(monthly_frame["as_of_date"].iloc[0]),
                    "negative_accounts": int((monthly_frame["volume"] < 0).sum()),
                    "raw_signed_balance": float(monthly_frame["volume"].sum()),
                    "paired_balance": float(selected["volume"].sum()),
                    "paired_model_npv": float(selected["model_npv"].sum()),
                }
            )
            output_parts.append(selected)
        output_records = pd.concat(output_parts, ignore_index=True).sort_values(["id", "period"])
        metadata: dict[str, object] = {
            "source_id": "uci-default-credit-card-clients-350",
            "source_url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "license": "CC BY 4.0",
            "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI. DOI:10.24432/C55S3H",
            "valuation_version": "assumed_one_month_receivable_v1",
            "target_metric": "model_npv",
            "currency": "TWD",
            "feature_names": ["volume", "pd"],
            "valuation_assumptions": dataclasses.asdict(self.valuation),
            "monthly_pd_by_status": {"-2,-1,0": 0.01, "1": 0.04, "2": 0.10, "3": 0.20, "4..9": 0.40},
            "raw_accounts": len(source_records),
            "paired_accounts": len(eligible_ids),
            "excluded_accounts": len(source_records) - len(eligible_ids),
            "periods": period_summaries,
            "limitations": [
                "Observed balances, assumed valuation; not a bank NPV or calibrated default model.",
                "Negative statements are excluded in either period; results cover only the paired subset.",
                "Nonpositive repayment codes are assigned low scenario PD; their risk is not estimated.",
                "Next-month default labels, demographics, and future monthly balances are not training factors.",
            ],
        }
        return output_records, metadata

    def save_dataset(
        self, source_path: pathlib.Path, output_path: pathlib.Path, initial_month: int, final_month: int
    ) -> pathlib.Path:
        metadata_path = output_path.with_suffix(".metadata.json")
        if output_path.exists() or metadata_path.exists():
            raise FileExistsError("Prepared UCI files already exist; choose a new output path")
        source_records = UciCreditReader().read_archive(source_path)
        prepared, metadata = self.prepare_periods(source_records, initial_month, final_month)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(output_path, index=False)
        metadata.update(source_sha256=file_checksum(source_path), dataset_sha256=file_checksum(output_path))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return output_path

    def _monthly_records(self, source_records: pd.DataFrame, calendar_month: int) -> pd.DataFrame:
        column_index = 10 - calendar_month
        status_column = "PAY_0" if column_index == 1 else f"PAY_{column_index}"
        month_end = calendar.monthrange(2005, calendar_month)[1]
        return pd.DataFrame(
            {
                "id": source_records["ID"].astype(str),
                "as_of_date": f"2005-{calendar_month:02d}-{month_end}",
                "volume": source_records[f"BILL_AMT{column_index}"].astype(float),
                "pd": self.valuation.default_probabilities(source_records[status_column]),
            }
        )
