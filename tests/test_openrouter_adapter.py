"""Exercise the actual provider SDK over an isolated HTTP transport, without external calls."""

import asyncio
import json

import httpx2
from pydantic import SecretStr

from ec_analysis.agents.providers import create_interpreter
from ec_analysis.config import AppSettings
from ec_analysis.schemas.state import AnalysisState


def test_real_sdk_contract_with_mocked_openrouter_http() -> None:
    requests = []

    def complete_request(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        payload = json.loads(request.content)
        requests.append(payload)
        tool_name = payload["tools"][0]["function"]["name"]
        arguments = dict(
            fact_ids=["change", "remainder", "reliability", "causality"], focus="summary", answer_supported=True
        )
        return httpx2.Response(
            200,
            json={
                "id": "mock-response",
                "object": "chat.completion",
                "provider": "MockProvider",
                "created": 0,
                "model": "mock/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240},
            },
        )

    async def verify_result() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(complete_request)) as http_client:
            service = create_interpreter(
                AppSettings(llm_model="mock/model", openrouter_api_key=SecretStr("local-test")), http_client
            )
            assert service is not None
            state = AnalysisState(
                initial_value=100, final_value=115, total_change=15, selected_model="ridge", unexplained_remainder=15
            )
            result = await service.explain_result(state, "Объясни изменение")
            assert result.checked and "остаток: 15" in result.text
            assert result.model_name == "openrouter:mock/model"

    asyncio.run(verify_result())
    assert len(requests) == 1
    assert "fact_ids" in requests[0]["tools"][0]["function"]["parameters"]["properties"]
