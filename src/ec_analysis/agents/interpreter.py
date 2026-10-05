"""Replaceable Pydantic AI model constrained to selecting verified analytical facts."""

from __future__ import annotations

import asyncio
import dataclasses
import typing

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ec_analysis.agents.evidence import EvidenceBuilder
from ec_analysis.agents.fact_checker import FactChecker
from ec_analysis.agents.prompts import INTERPRETER_INSTRUCTIONS
from ec_analysis.schemas.interpretation import CheckedExplanation, EvidenceCatalog, ExplanationPlan
from ec_analysis.schemas.state import AnalysisState


def build_interpreter_agent(model_name: str | Model) -> Agent[EvidenceCatalog, ExplanationPlan]:
    interpreter_agent = Agent(
        model_name,
        deps_type=EvidenceCatalog,
        output_type=ExplanationPlan,
        instructions=INTERPRETER_INSTRUCTIONS,
        retries=2,
    )

    @interpreter_agent.instructions
    def provide_evidence(context: RunContext[EvidenceCatalog]) -> str:
        required_identifiers = [fact.identifier for fact in context.deps.facts if fact.required]
        return (
            "Every answer MUST include these exact fact_ids, including narrow or unsupported questions: "
            + ", ".join(required_identifiers)
            + ". If answer_supported=false, also include unsupported.\nVerified evidence catalog:\n"
            + context.deps.model_dump_json()
        )

    @interpreter_agent.output_validator
    def verify_output(context: RunContext[EvidenceCatalog], output: ExplanationPlan) -> ExplanationPlan:
        try:
            return FactChecker().validate_plan(context.deps, output)
        except ValueError as failure:
            raise ModelRetry(str(failure)) from failure

    return interpreter_agent


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class InterpretationService:
    agent: Agent[EvidenceCatalog, ExplanationPlan]
    model_name: str
    timeout_seconds: float = 60
    fact_checker: FactChecker = dataclasses.field(default_factory=FactChecker)
    evidence_builder: EvidenceBuilder = dataclasses.field(default_factory=EvidenceBuilder)

    async def explain_result(self, state: AnalysisState, question: str) -> CheckedExplanation:
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4000 characters")
        catalog = self.evidence_builder.build_catalog(self.fact_checker.validate_state(state))
        async with asyncio.timeout(self.timeout_seconds):
            completed = await self.agent.run(
                question,
                deps=catalog,
                usage_limits=UsageLimits(request_limit=3, total_tokens_limit=25000),
            )
        return self.fact_checker.render_explanation(catalog, completed.output, self.model_name)
