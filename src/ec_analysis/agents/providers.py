"""Provider configuration only; the numerical method has no provider dependency."""

from __future__ import annotations

import httpx2
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from ec_analysis.agents.interpreter import InterpretationService, build_interpreter_agent
from ec_analysis.config import AppSettings


def create_interpreter(settings: AppSettings, http_client: httpx2.AsyncClient) -> InterpretationService | None:
    if (
        not settings.llm_model
        or settings.openrouter_api_key is None
        or not settings.openrouter_api_key.get_secret_value()
    ):
        return None
    model = OpenRouterModel(
        settings.llm_model,
        provider=OpenRouterProvider(
            api_key=settings.openrouter_api_key.get_secret_value(),
            http_client=http_client,
            app_title="EC factor analysis",
        ),
    )
    return InterpretationService(
        agent=build_interpreter_agent(model),
        model_name=f"openrouter:{settings.llm_model}",
        timeout_seconds=settings.llm_timeout_seconds,
    )
