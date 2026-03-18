from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
import pytest

from tests.chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    build_request_headers,
    get_api_key,
    request_response,
    resolve_base_url,
)


COMPLETIONS_PATH = "completions"
MODELS_PATH = "models"
DEFAULT_MAX_TOKENS = 16
LIGHT_RATE_LIMIT_PROBE_CONCURRENCY = 40

pytestmark = pytest.mark.api_compatibility


def build_chat_payload(
    model: str,
    *,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": max_completion_tokens,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
    }
    if model == "qwen35":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def build_raw_completion_payload(model: str) -> dict[str, object]:
    return {
        "model": model,
        "prompt": "Reply with the single word quartz.",
        "temperature": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }


def build_forced_tool_choice_payload() -> dict[str, object]:
    return {
        "model": "minimax-m25",
        "temperature": 0,
        "max_completion_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Call the Echo tool once with word=ping."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Echo",
                    "description": "Echo input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                        },
                        "required": ["word"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "Echo"}},
    }


def assert_usage_arithmetic(payload: Mapping[str, object]) -> None:
    usage = payload.get("usage")
    assert isinstance(usage, Mapping), payload

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    assert isinstance(prompt_tokens, int), usage
    assert isinstance(completion_tokens, int), usage
    assert isinstance(total_tokens, int), usage
    assert prompt_tokens + completion_tokens == total_tokens, usage


def assert_openai_error_shape(payload: Mapping[str, object]) -> None:
    error = payload.get("error")
    assert isinstance(error, Mapping), payload
    assert isinstance(error.get("message"), str) and error["message"].strip(), error
    assert isinstance(error.get("type"), str) and error["type"].strip(), error


def find_model_card(payload: Mapping[str, object], model: str) -> Mapping[str, object]:
    data = payload.get("data")
    assert isinstance(data, list) and data, payload

    for item in data:
        if isinstance(item, Mapping) and item.get("id") == model:
            return item

    raise AssertionError(f"/v1/models did not report model {model!r}: {payload}")


def request_with_client(
    client: httpx.Client,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    recorder: FailureArtifactRecorder | None = None,
) -> httpx.Response:
    request_kwargs: dict[str, object] = {}
    request_payload = payload or {}
    if payload is not None:
        request_kwargs["json"] = payload

    request = client.build_request(method, path, **request_kwargs)
    response: httpx.Response | None = None
    try:
        response = client.send(request)
    except Exception as exc:
        if recorder is not None:
            recorder.add_http_exchange(
                request=request,
                response=None,
                request_payload=request_payload,
                exception=exc,
            )
        raise

    if recorder is not None:
        recorder.add_http_exchange(
            request=request,
            response=response,
            request_payload=request_payload,
        )
    return response


def test_chat_completions_returns_openai_shape_and_usage(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(
        http_client,
        "POST",
        CHAT_COMPLETIONS_PATH,
        build_chat_payload(model),
        recorder=failure_artifact_recorder,
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"]

    choices = payload["choices"]
    assert isinstance(choices, list) and choices
    message = choices[0]["message"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)
    assert message["content"].strip()
    assert_usage_arithmetic(payload)


def test_raw_completions_returns_openai_shape_and_usage(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(
        http_client,
        "POST",
        COMPLETIONS_PATH,
        build_raw_completion_payload(model),
        recorder=failure_artifact_recorder,
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["object"] == "text_completion"
    assert payload["model"]

    choices = payload["choices"]
    assert isinstance(choices, list) and choices
    first_choice = choices[0]
    assert isinstance(first_choice.get("text"), str)
    assert first_choice["text"].strip()
    assert_usage_arithmetic(payload)


def test_models_endpoint_lists_selected_model(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(
        http_client,
        "GET",
        MODELS_PATH,
        recorder=failure_artifact_recorder,
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["object"] == "list"

    card = find_model_card(payload, model)
    assert isinstance(card.get("object"), str) and card["object"]
    assert isinstance(card.get("created"), int)
    assert isinstance(card.get("owned_by"), str) and card["owned_by"]


def test_chat_completions_bad_request_returns_openai_error_shape(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(
        http_client,
        "POST",
        CHAT_COMPLETIONS_PATH,
        {"model": model, "messages": "hi"},
        recorder=failure_artifact_recorder,
    )
    assert response.status_code == 400, response.text

    payload = response.json()
    assert_openai_error_shape(payload)


def test_chat_completions_invalid_api_key_returns_openai_error_shape(
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    if not get_api_key():
        pytest.skip("OPENAI_API_KEY is empty; unauthorized-path verification is not meaningful.")

    headers = build_request_headers()
    headers["Authorization"] = "Bearer invalid-key-for-test"

    with httpx.Client(
        base_url=resolve_base_url(),
        headers=headers,
        timeout=httpx.Timeout(60.0),
    ) as client:
        response = request_with_client(
            client,
            "POST",
            CHAT_COMPLETIONS_PATH,
            build_chat_payload(model),
            recorder=failure_artifact_recorder,
        )

    assert response.status_code == 401, response.text
    payload = response.json()
    assert_openai_error_shape(payload)


def test_unknown_route_returns_openai_error_shape(
    http_client: httpx.Client,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(
        http_client,
        "POST",
        "__codex_not_found__/chat/completions",
        {"foo": "bar"},
        recorder=failure_artifact_recorder,
    )
    assert response.status_code == 404, response.text

    payload = response.json()
    assert_openai_error_shape(payload)


class TestGLM5RateLimitProbe:
    __test__ = True
    MODEL_NAME = "glm5"

    def test_light_rate_limit_probe_returns_only_200_or_429(
        self,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        async def run_probe() -> list[httpx.Response]:
            async with httpx.AsyncClient(
                base_url=resolve_base_url(),
                headers=build_request_headers(),
                timeout=httpx.Timeout(30.0),
            ) as client:
                tasks = [
                    client.post(
                        CHAT_COMPLETIONS_PATH,
                        json=build_chat_payload(self.MODEL_NAME, max_completion_tokens=1),
                    )
                    for _ in range(LIGHT_RATE_LIMIT_PROBE_CONCURRENCY)
                ]
                return await asyncio.gather(*tasks)

        responses = asyncio.run(run_probe())

        for response in responses:
            failure_artifact_recorder.add_http_exchange(
                request=response.request,
                response=response,
                request_payload=build_chat_payload(self.MODEL_NAME, max_completion_tokens=1),
            )

            assert response.status_code in {200, 429}, response.text
            if response.status_code == 429:
                payload = response.json()
                assert_openai_error_shape(payload)


class TestMinimaxM25ApiCompatibility:
    __test__ = True
    MODEL_NAME = "minimax-m25"

    def test_forced_named_tool_choice_returns_openai_error_shape_when_upstream_500_reproduces(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response = request_response(
            http_client,
            "POST",
            CHAT_COMPLETIONS_PATH,
            build_forced_tool_choice_payload(),
            recorder=failure_artifact_recorder,
        )

        if response.status_code == 200:
            pytest.skip("Known minimax-m25 forced tool-choice 500 path did not reproduce in this run.")

        assert response.status_code == 500, response.text
        payload = response.json()
        assert_openai_error_shape(payload)
