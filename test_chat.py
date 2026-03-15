from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    request_json,
    request_sse,
)


def build_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "collect_weather_args",
                "description": "Collect weather lookup arguments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_chat_template_disable_thinking() -> dict[str, object]:
    return {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
        }
    }


class BaseHTTPXChatTests:
    __test__ = False

    MODEL_NAME: str = ""
    TOOL_REQUEST_MODE = "forced_named_tool_choice"
    RESPONSE_FORMAT_CHANNEL = "content"

    def create_request_overrides(self) -> Mapping[str, object]:
        return {}

    def tool_request_overrides(self) -> Mapping[str, object]:
        return {}

    def response_format_request_overrides(self) -> Mapping[str, object]:
        return {}

    def build_create_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
        }
        payload.update(self.create_request_overrides())
        return payload

    def build_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
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
        }
        if self.TOOL_REQUEST_MODE == "forced_named_tool_choice":
            payload["tool_choice"] = {"type": "function", "function": {"name": "collect_weather_args"}}
        elif self.TOOL_REQUEST_MODE == "auto_tool_choice":
            payload["tool_choice"] = "auto"
        payload.update(self.tool_request_overrides())
        return payload

    def build_stream_payload(self) -> dict[str, object]:
        return {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the word quartz."},
            ],
        }

    def build_response_format_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
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
        }
        payload.update(self.response_format_request_overrides())
        return payload

    def extract_structured_text(self, message: Mapping[str, Any]) -> str:
        channel = self.RESPONSE_FORMAT_CHANNEL
        value = message.get(channel)
        assert isinstance(value, str), f"Expected structured JSON in message.{channel}, got {value!r}"
        return value

    def test_create_returns_non_empty_assistant_message(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_create_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert message["content"].strip()
        assert completion["usage"]["total_tokens"] > 0

    def test_create_returns_tool_call(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_calls = message["tool_calls"]

        assert tool_calls
        tool_call = tool_calls[0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "collect_weather_args"

        arguments = json.loads(tool_call["function"]["arguments"])
        assert arguments["city"].lower().startswith("tokyo")
        assert arguments["unit"] == "celsius"

    def test_stream_sse_emits_content_and_done(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        parts: list[str] = []
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_stream_payload(),
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

    def test_response_format_returns_structured_output(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_response_format_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        parsed = json.loads(self.extract_structured_text(message))

        assert set(parsed) == {"word", "length"}
        assert parsed["word"].lower() == "ping"
        assert parsed["length"] == 4


class BaseOpenAICompatibleChatTests(BaseHTTPXChatTests):
    __test__ = False


class BaseThinkingDisabledStructuredOutputTests(BaseHTTPXChatTests):
    __test__ = False

    TOOL_REQUEST_MODE = "auto_tool_choice"

    def response_format_request_overrides(self) -> Mapping[str, object]:
        return build_chat_template_disable_thinking()


class BaseReasoningStructuredOutputTests(BaseHTTPXChatTests):
    __test__ = False

    TOOL_REQUEST_MODE = "auto_tool_choice"
    RESPONSE_FORMAT_CHANNEL = "reasoning"


class TestKimiK25ChatCompletions(BaseOpenAICompatibleChatTests):
    __test__ = True
    MODEL_NAME = "kimi-k25"


class TestGLM5ChatCompletions(BaseThinkingDisabledStructuredOutputTests):
    __test__ = True
    MODEL_NAME = "glm5"


class TestQwen35ChatCompletions(BaseThinkingDisabledStructuredOutputTests):
    __test__ = True
    MODEL_NAME = "qwen35"


class TestMinimaxM25ChatCompletions(BaseReasoningStructuredOutputTests):
    __test__ = True
    MODEL_NAME = "minimax-m25"


class TestMinimaxM21ChatCompletions(BaseReasoningStructuredOutputTests):
    __test__ = True
    MODEL_NAME = "minimax-m21"
