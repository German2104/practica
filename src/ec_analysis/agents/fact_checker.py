"""Structural checks before an analysis state is exposed to the LLM."""

from __future__ import annotations

import typing

from ec_analysis.schemas.interpretation import CheckedExplanation, CheckedParagraph, EvidenceCatalog, ExplanationPlan
from ec_analysis.schemas.state import AnalysisState


@typing.final
class FactChecker:
    def validate_state(self, analysis_state: AnalysisState) -> AnalysisState:
        """Reject an inconsistent numerical state before building evidence."""
        return AnalysisState.model_validate(analysis_state.model_dump())

    def validate_plan(self, catalog: EvidenceCatalog, plan: ExplanationPlan) -> ExplanationPlan:
        fact_map = {fact.identifier: fact for fact in catalog.facts}
        if len(plan.fact_ids) != len(set(plan.fact_ids)) or any(name not in fact_map for name in plan.fact_ids):
            raise ValueError("Explanation contains duplicate or unknown evidence identifiers")
        required_ids = {fact.identifier for fact in catalog.facts if fact.required}
        missing_ids = sorted(required_ids.difference(plan.fact_ids))
        if missing_ids:
            raise ValueError("Explanation omitted required evidence identifiers: " + ", ".join(missing_ids))
        if not plan.answer_supported and "unsupported" not in plan.fact_ids:
            raise ValueError("An unsupported question must include its explicit limitation")
        if any(fact_map[name].category == "importance" and name != "importance-method" for name in plan.fact_ids):
            if "importance-method" not in plan.fact_ids:
                raise ValueError("Global importance must include the distinction from portfolio contribution")
        return plan

    def render_explanation(
        self,
        catalog: EvidenceCatalog,
        plan: ExplanationPlan,
        model_name: str,
    ) -> CheckedExplanation:
        checked_plan = self.validate_plan(catalog, plan)
        fact_map = {fact.identifier: fact for fact in catalog.facts}
        paragraphs = [
            CheckedParagraph(
                fact_id=identifier,
                text=fact_map[identifier].text,
                references=fact_map[identifier].references,
            )
            for identifier in checked_plan.fact_ids
        ]
        return CheckedExplanation(
            analysis_sha256=catalog.analysis_sha256,
            model_name=model_name,
            answer_supported=checked_plan.answer_supported,
            paragraphs=paragraphs,
            text="\n\n".join(paragraph.text for paragraph in paragraphs),
        )
