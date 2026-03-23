from __future__ import annotations

from collections.abc import Mapping

import httpx
import pytest

from tests.chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    request_response,
)


COMPLETIONS_PATH = "completions"
MODELS_PATH = "models"
DEFAULT_MAX_TOKENS = 16

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


def find_model_card(payload: Mapping[str, object], model: str) -> Mapping[str, object]:
    data = payload.get("data")
    assert isinstance(data, list) and data, payload

    for item in data:
        if isinstance(item, Mapping) and item.get("id") == model:
            return item

    raise AssertionError(f"/v1/models did not report model {model!r}: {payload}")


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
