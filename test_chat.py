from __future__ import annotations

import json
from collections.abc import Mapping

import httpx

from chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    collect_stream_text,
    request_json,
    request_sse,
)


def build_structured_output_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "StructuredOutput",
                "description": "Return the final structured answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "length": {"type": "integer"},
                    },
                    "required": ["word", "length"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def extract_first_tool_call(message: Mapping[str, object]) -> dict[str, object]:
    tool_calls = message["tool_calls"]
    assert isinstance(tool_calls, list) and tool_calls
    tool_call = tool_calls[0]
    assert isinstance(tool_call, dict)
    return tool_call


def parse_tool_arguments(tool_call: Mapping[str, object]) -> dict[str, object]:
    arguments = json.loads(tool_call["function"]["arguments"])
    assert isinstance(arguments, dict)
    return arguments


class BaseHTTPXChatTests:
    __test__ = False

    MODEL_NAME: str = ""
    TOOL_REQUEST_MODE = "forced_named_tool_choice"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED: bool | None = None

    def create_request_overrides(self) -> Mapping[str, object]:
        return {}

    def tool_request_overrides(self) -> Mapping[str, object]:
        return {}

    def apply_tool_choice(self, payload: dict[str, object], tool_name: str) -> None:
        if self.TOOL_REQUEST_MODE == "forced_named_tool_choice":
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        elif self.TOOL_REQUEST_MODE == "auto_tool_choice":
            payload["tool_choice"] = "auto"

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

    def build_disable_thinking_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the single word quartz."},
            ],
        }
        payload.update(self.create_request_overrides())
        return payload

    def build_disable_thinking_stream_payload(self) -> dict[str, object]:
        payload = self.build_disable_thinking_payload()
        payload["stream"] = True
        return payload

    def build_structured_output_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_structured_output_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the StructuredOutput tool to return a JSON object where "
                        "the word is exactly 'ping' and the length is exactly 4. "
                        "Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "StructuredOutput")
        payload.update(self.tool_request_overrides())
        return payload

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

    def test_stream_sse_emits_content_and_done(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_stream_payload(),
            recorder=failure_artifact_recorder,
        )
        stream_result = collect_stream_text(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert stream_result.chunk_count > 0
        assert stream_result.saw_done
        assert stream_result.text
        assert "quartz" in stream_result.text.lower()

    def test_create_accepts_chat_template_kwargs_enable_thinking_false(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_disable_thinking_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]

        assert completion["object"] == "chat.completion"
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert "quartz" in message["content"].lower()

        reasoning = message.get("reasoning")
        if reasoning is not None:
            assert isinstance(reasoning, str)
        if self.EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED is True:
            assert reasoning is None

    def test_stream_accepts_chat_template_kwargs_enable_thinking_false(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_disable_thinking_stream_payload(),
            recorder=failure_artifact_recorder,
        )
        stream_result = collect_stream_text(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert stream_result.chunk_count > 0
        assert stream_result.saw_done
        assert stream_result.text
        assert "quartz" in stream_result.text.lower()
        if self.EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED is True:
            assert stream_result.reasoning is None

    def test_structured_output_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_structured_output_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "StructuredOutput"

        parsed = parse_tool_arguments(tool_call)
        assert set(parsed) == {"word", "length"}
        assert parsed["word"].lower() == "ping"
        assert parsed["length"] == 4


class BaseOpenAICompatibleChatTests(BaseHTTPXChatTests):
    __test__ = False


class BaseRelaxedToolChoiceChatTests(BaseHTTPXChatTests):
    __test__ = False

    TOOL_REQUEST_MODE = "auto_tool_choice"


class TestKimiK25ChatCompletions(BaseOpenAICompatibleChatTests):
    __test__ = True
    MODEL_NAME = "kimi-k25"


class TestGLM5ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "glm5"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED = True


class TestQwen35ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "qwen35"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED = True


class TestMinimaxM25ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "minimax-m25"


class TestMinimaxM21ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "minimax-m21"
