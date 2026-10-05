"""AG-UI events emitted only after a complete explanation passes its evidence checks."""

from __future__ import annotations

import dataclasses
import typing
import uuid

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded

from ec_analysis.agents.interpreter import InterpretationService
from ec_analysis.schemas.state import AnalysisState


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class CheckedAGUIStream:
    interpreter: InterpretationService

    async def stream_events(
        self,
        run_input: RunAgentInput,
        analysis_state: AnalysisState,
        question: str,
    ) -> typing.AsyncIterator[str]:
        encoder = EventEncoder()
        yield encoder.encode(RunStartedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
        try:
            checked_result = await self.interpreter.explain_result(analysis_state, question)
        except (ValueError, TimeoutError, ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded):
            yield encoder.encode(
                RunErrorEvent(
                    message="Ответ не прошёл проверку или модель недоступна. Проверьте подключение и повторите вопрос.",
                    code="EXPLANATION_NOT_VERIFIED",
                )
            )
            return
        message_identifier = uuid.uuid4().hex
        yield encoder.encode(StateSnapshotEvent(snapshot={"explanation": checked_result.model_dump(mode="json")}))
        yield encoder.encode(TextMessageStartEvent(message_id=message_identifier))
        for paragraph in checked_result.paragraphs:
            yield encoder.encode(TextMessageContentEvent(message_id=message_identifier, delta=paragraph.text + "\n\n"))
        yield encoder.encode(TextMessageEndEvent(message_id=message_identifier))
        yield encoder.encode(RunFinishedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
