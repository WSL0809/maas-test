from __future__ import annotations

import json

import httpx
import pytest

from chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    request_json,
    request_sse,
)
from test_chat import build_tool_definition


pytestmark = pytest.mark.strict_compat


def test_create_returns_non_empty_assistant_message(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    completion = request_json(
        http_client,
        CHAT_COMPLETIONS_PATH,
        {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
        },
        recorder=failure_artifact_recorder,
    )

    message = completion["choices"][0]["message"]

    assert completion["object"] == "chat.completion"
    assert completion["model"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)
    assert message["content"].strip()
    assert completion["usage"]["total_tokens"] > 0


def test_create_with_forced_tool_choice_returns_tool_call(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    completion = request_json(
        http_client,
        CHAT_COMPLETIONS_PATH,
        {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tool_choice": {"type": "function", "function": {"name": "collect_weather_args"}},
            "tools": build_tool_definition(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Use the provided function to submit these exact arguments: "
                        "city='Tokyo', unit='celsius'. Do not answer with plain text."
                    ),
                },
            ],
        },
        recorder=failure_artifact_recorder,
    )

    tool_calls = completion["choices"][0]["message"]["tool_calls"]

    assert tool_calls
    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "collect_weather_args"

    arguments = json.loads(tool_call["function"]["arguments"])
    assert arguments["city"].lower().startswith("tokyo")
    assert arguments["unit"] == "celsius"


def test_stream_sse_emits_content_and_done(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    parts: list[str] = []
    response, events, raw_text = request_sse(
        http_client,
        CHAT_COMPLETIONS_PATH,
        {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the word quartz."},
            ],
        },
        recorder=failure_artifact_recorder,
    )
    chunk_count = 0
    saw_done = False

    assert response.status_code == 200, raw_text
    assert response.headers["content-type"].startswith("text/event-stream")

    for event in events:
        if event == "[DONE]":
            saw_done = True
            continue

        chunk_count += 1
        chunk = json.loads(event)
        if not chunk.get("choices"):
            continue

        delta = chunk["choices"][0].get("delta", {}).get("content")
        if delta:
            parts.append(delta)

    text = "".join(parts).strip()

    assert chunk_count > 0
    assert saw_done
    assert text
    assert "quartz" in text.lower()


def test_response_format_json_schema_returns_valid_json(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    completion = request_json(
        http_client,
        CHAT_COMPLETIONS_PATH,
        {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "parsed_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "length": {"type": "integer"},
                        },
                        "required": ["word", "length"],
                        "additionalProperties": False,
                    },
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Return a structured answer where the word is exactly 'ping' "
                        "and the length is exactly 4."
                    ),
                },
            ],
        },
        recorder=failure_artifact_recorder,
    )

    message = completion["choices"][0]["message"]
    parsed = json.loads(message["content"])

    assert isinstance(message["content"], str)
    assert set(parsed) == {"word", "length"}
    assert parsed["word"].lower() == "ping"
    assert parsed["length"] == 4
