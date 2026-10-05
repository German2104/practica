"""The LLM selects audited facts; it cannot introduce new text, numbers or claims."""

from __future__ import annotations

import typing

from pydantic import BaseModel, ConfigDict, Field


class EvidenceFact(BaseModel):
    identifier: str
    text: str
    references: list[str]
    category: str
    required: bool = False


class EvidenceCatalog(BaseModel):
    analysis_sha256: str
    target_metric: str
    facts: list[EvidenceFact]


class ExplanationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_ids: list[str] = Field(min_length=1, max_length=40)
    focus: typing.Literal["summary", "factors", "interactions", "quality", "shapes", "importance", "limitations"]
    answer_supported: bool


class CheckedParagraph(BaseModel):
    fact_id: str
    text: str
    references: list[str]


class CheckedExplanation(BaseModel):
    method: str = "verified_fact_selection_v1"
    analysis_sha256: str
    model_name: str
    answer_supported: bool
    paragraphs: list[CheckedParagraph]
    text: str
    checked: typing.Literal[True] = True
