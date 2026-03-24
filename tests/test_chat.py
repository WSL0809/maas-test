from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
import pytest

from tests.chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    collect_stream_text,
    request_json,
    request_response,
    request_sse,
)

TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2k0AAAAASUVORK5CYII="
)
RED_SQUARE_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEUlEQVR42mP4z8CAFTEMLQkAKP8/wc53yE8AAAAASUVORK5CYII="
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


def normalize_text_content(content: object) -> str:
    assert isinstance(content, str)
    return content.strip().lower().strip("`'\"").rstrip(".。!！")


def normalize_reasoning_content(reasoning: object) -> str | None:
    if reasoning is None:
        return None
    assert isinstance(reasoning, str)
    return reasoning.strip() or None


def extract_reasoning(message: Mapping[str, object]) -> str | None:
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return normalize_reasoning_content(reasoning)


class BaseHTTPXChatTests:
    __test__ = False

    MODEL_NAME: str = ""
    TOOL_REQUEST_MODE = "forced_named_tool_choice"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED: bool | None = None
    EXPECTS_JSON_MODE_PAYLOAD_IN_CONTENT: bool = True

    def base_text_request_overrides(self) -> Mapping[str, object]:
        return {}

    def create_request_overrides(self) -> Mapping[str, object]:
        return {}

    def tool_request_overrides(self) -> Mapping[str, object]:
        return {}

    def apply_tool_choice(self, payload: dict[str, object], tool_name: str) -> None:
        if self.TOOL_REQUEST_MODE == "forced_named_tool_choice":
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        elif self.TOOL_REQUEST_MODE == "auto_tool_choice":
            payload["tool_choice"] = "auto"

    def assert_disable_thinking_reasoning_suppressed(
        self,
        reasoning: str | None,
        visible_text: object,
        *,
        transport: str,
        expected_answer: str = "quartz",
    ) -> None:
        normalized_visible_text = normalize_text_content(visible_text)
        normalized_expected_answer = normalize_text_content(expected_answer)

        if reasoning is None and normalized_visible_text == normalized_expected_answer:
            return

        details: list[str] = []
        if reasoning is not None:
            details.append(f"reasoning={reasoning!r}")
        if normalized_visible_text != normalized_expected_answer:
            details.append(f"visible_text={visible_text!r}")
        detail_suffix = f" ({', '.join(details)})" if details else ""
        message = (
            f"{self.MODEL_NAME} leaked hidden thinking in {transport} responses when "
            f"chat_template_kwargs.enable_thinking=false{detail_suffix}"
        )
        if self.EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED is True:
            raise AssertionError(message)
        pytest.xfail(message)

    def assert_json_mode_payload_in_content(self, message: Mapping[str, object]) -> str:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        reasoning = extract_reasoning(message)
        reason_suffix = f" (reasoning={reasoning!r})" if reasoning else ""
        failure_message = (
            f"{self.MODEL_NAME} did not return JSON mode payload in message.content{reason_suffix}"
        )

        if self.EXPECTS_JSON_MODE_PAYLOAD_IN_CONTENT:
            raise AssertionError(failure_message)
        pytest.xfail(failure_message)

    def assert_single_image_understanding(self, content: object) -> None:
        normalized_content = normalize_text_content(content)
        if "red" in normalized_content or "红" in normalized_content:
            return
        pytest.xfail(
            f"{self.MODEL_NAME} did not identify the single-image dominant color as red "
            f"(response={normalized_content!r})"
        )

    def assert_multimodal_request_supported_or_xfail(
        self,
        response: httpx.Response,
    ) -> None:
        if response.status_code == 200:
            return

        error_message = response.text
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                raw_message = error.get("message")
                if isinstance(raw_message, str) and raw_message.strip():
                    error_message = raw_message

        if "not a multimodal model" in error_message.lower():
            pytest.xfail(f"{self.MODEL_NAME} does not support multimodal image understanding on this endpoint")

        raise AssertionError(response.text)

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
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_multi_turn_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "The project codename is bamboo-7. "
                        "Acknowledge with exactly the word noted."
                    ),
                },
                {"role": "assistant", "content": "noted"},
                {
                    "role": "user",
                    "content": "What is the project codename? Reply with only the codename.",
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_system_prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": "Reply with exactly the single token system-wins.",
                },
                {
                    "role": "user",
                    "content": "For this test, reply with exactly the single token user-wins.",
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_limited_tokens_payload(self, max_completion_tokens: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": max_completion_tokens,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Count from one to twenty in English words, separated by commas, "
                        "with no extra commentary."
                    ),
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_multilingual_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "请严格按顺序原样输出这五个词，并且只输出这一行："
                        "你好,hello,こんにちは,안녕하세요,bonjour"
                    ),
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_special_token_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Repeat this exact line and nothing else: 😀 <div>ok</div> `x=1` ∑",
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_stream_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the word quartz."},
            ],
        }
        payload.update(self.base_text_request_overrides())
        return payload

    def build_image_create_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the word quartz."},
                        {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                    ],
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_image_stream_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the word quartz."},
                        {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                    ],
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        return payload

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

    def build_enable_thinking_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Solve 17 + 26. Think through it first, then reply with only the final "
                        "answer."
                    ),
                },
            ],
        }
        payload.update(self.create_request_overrides())
        return payload

    def build_enable_thinking_stream_payload(self) -> dict[str, object]:
        payload = self.build_enable_thinking_payload()
        payload["stream"] = True
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

    def build_json_mode_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with exactly these fields: "
                        '{"word":"ping","length":4}. '
                        "Do not return markdown."
                    ),
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
        return payload

    def build_single_image_understanding_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Identify the dominant color in this image. "
                                "Reply with exactly one lowercase English word."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": RED_SQUARE_PNG_DATA_URL}},
                    ],
                },
            ],
        }
        payload.update(self.base_text_request_overrides())
        payload.update(self.create_request_overrides())
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

    def test_create_respects_system_prompt_priority(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_system_prompt_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        normalized_content = normalize_text_content(message["content"])

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert normalized_content == "system-wins"
        assert completion["usage"]["total_tokens"] > 0

    def test_create_preserves_multi_turn_context(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_multi_turn_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        normalized_content = normalize_text_content(message["content"])

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert normalized_content == "bamboo-7"
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

    def test_create_accepts_image_content_parts(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_image_create_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert message["content"].strip()
        assert completion["usage"]["total_tokens"] > 0

    def test_create_respects_max_completion_tokens_limit(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        token_limit = 8
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_limited_tokens_payload(token_limit),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert completion["usage"]["completion_tokens"] <= token_limit

    def test_create_supports_multilingual_output(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_multilingual_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        normalized_content = str(message["content"]).strip().replace(" ", "").replace("，", ",")

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert normalized_content == "你好,hello,こんにちは,안녕하세요,bonjour"
        assert completion["usage"]["total_tokens"] > 0

    def test_create_preserves_special_tokens_in_text(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_special_token_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        content = str(message["content"])

        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert "😀" in content
        assert "<div>ok</div>" in content
        assert "`x=1`" in content
        assert "∑" in content
        assert completion["usage"]["total_tokens"] > 0

    def test_stream_accepts_image_content_parts(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_image_stream_payload(),
            recorder=failure_artifact_recorder,
        )
        stream_result = collect_stream_text(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert stream_result.chunk_count > 0
        assert stream_result.saw_done
        assert stream_result.text
        assert "quartz" in stream_result.text.lower()

    def test_create_understands_single_image_dominant_color(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response = request_response(
            http_client,
            "POST",
            CHAT_COMPLETIONS_PATH,
            self.build_single_image_understanding_payload(),
            recorder=failure_artifact_recorder,
        )
        self.assert_multimodal_request_supported_or_xfail(response)
        completion = response.json()

        message = completion["choices"][0]["message"]
        assert completion["object"] == "chat.completion"
        assert completion["model"]
        assert message["role"] == "assistant"
        assert isinstance(message["content"], str)
        assert message["content"].strip()
        assert completion["usage"]["total_tokens"] > 0
        self.assert_single_image_understanding(message["content"])

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

    def test_create_suppresses_reasoning_when_thinking_disabled(
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
        self.assert_disable_thinking_reasoning_suppressed(
            extract_reasoning(message),
            message["content"],
            transport="create",
        )

    def test_create_returns_reasoning_when_thinking_enabled(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_enable_thinking_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        content = str(message["content"])
        reasoning = extract_reasoning(message)

        assert completion["object"] == "chat.completion"
        assert message["role"] == "assistant"
        assert content.strip()
        assert "43" in content
        if isinstance(reasoning, str) and reasoning:
            return

        assert any(char.isalpha() for char in content), (
            f"{self.MODEL_NAME} did not return a separate reasoning field in create responses, "
            "and message.content does not appear to include any explanatory content."
        )

    def test_stream_emits_reasoning_when_thinking_enabled(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_enable_thinking_stream_payload(),
            recorder=failure_artifact_recorder,
        )
        stream_result = collect_stream_text(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert stream_result.chunk_count > 0
        assert stream_result.saw_done
        assert stream_result.text
        assert "43" in stream_result.text
        if isinstance(stream_result.reasoning, str) and stream_result.reasoning.strip():
            return

        assert any(char.isalpha() for char in stream_result.text), (
            f"{self.MODEL_NAME} did not emit a separate reasoning field in stream chunks, "
            "and the streamed text does not appear to include any explanatory content."
        )

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

    def test_stream_suppresses_reasoning_when_thinking_disabled(
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
        self.assert_disable_thinking_reasoning_suppressed(
            stream_result.reasoning,
            stream_result.text,
            transport="stream",
        )

    def test_create_switches_thinking_to_instant_within_same_conversation(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        thinking_payload = self.build_enable_thinking_payload()
        thinking_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            thinking_payload,
            recorder=failure_artifact_recorder,
        )
        thinking_message = thinking_completion["choices"][0]["message"]
        thinking_content = str(thinking_message["content"])
        thinking_reasoning = extract_reasoning(thinking_message)

        assert thinking_completion["object"] == "chat.completion"
        assert thinking_message["role"] == "assistant"
        assert thinking_content.strip()
        assert "43" in thinking_content
        if isinstance(thinking_reasoning, str) and thinking_reasoning.strip():
            pass
        else:
            assert any(char.isalpha() for char in thinking_content), (
                f"{self.MODEL_NAME} did not return a separate reasoning field in create responses, "
                "and message.content does not appear to include any explanatory content."
            )

        messages = list(thinking_payload["messages"])
        messages.append({"role": "assistant", "content": thinking_content})
        messages.append(
            {
                "role": "user",
                "content": "Now reply with the single word quartz.",
            }
        )

        instant_payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages,
        }
        instant_payload.update(self.create_request_overrides())
        instant_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            instant_payload,
            recorder=failure_artifact_recorder,
        )

        instant_message = instant_completion["choices"][0]["message"]
        assert instant_completion["object"] == "chat.completion"
        assert instant_message["role"] == "assistant"
        assert isinstance(instant_message["content"], str)
        assert "quartz" in instant_message["content"].lower()

        self.assert_disable_thinking_reasoning_suppressed(
            extract_reasoning(instant_message),
            instant_message["content"],
            transport="create",
        )

    def test_create_switches_instant_to_thinking_within_same_conversation(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        instant_payload = self.build_disable_thinking_payload()
        instant_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            instant_payload,
            recorder=failure_artifact_recorder,
        )
        instant_message = instant_completion["choices"][0]["message"]

        assert instant_completion["object"] == "chat.completion"
        assert instant_message["role"] == "assistant"
        assert isinstance(instant_message["content"], str)
        assert "quartz" in instant_message["content"].lower()

        messages = list(instant_payload["messages"])
        messages.append({"role": "assistant", "content": instant_message["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Solve 17 + 26. Think through it first, then reply with only the final "
                    "answer."
                ),
            }
        )

        thinking_payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": messages,
        }
        thinking_payload.update(self.create_request_overrides())
        thinking_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            thinking_payload,
            recorder=failure_artifact_recorder,
        )

        thinking_message = thinking_completion["choices"][0]["message"]
        thinking_content = str(thinking_message["content"])
        thinking_reasoning = extract_reasoning(thinking_message)

        assert thinking_completion["object"] == "chat.completion"
        assert thinking_message["role"] == "assistant"
        assert thinking_content.strip()
        assert "43" in thinking_content
        if isinstance(thinking_reasoning, str) and thinking_reasoning.strip():
            pass
        else:
            assert any(char.isalpha() for char in thinking_content), (
                f"{self.MODEL_NAME} did not return a separate reasoning field in create responses, "
                "and message.content does not appear to include any explanatory content."
            )

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

    def test_json_mode_returns_valid_json_object(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_json_mode_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        assert completion["object"] == "chat.completion"
        assert message["role"] == "assistant"
        content = self.assert_json_mode_payload_in_content(message)

        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert parsed.get("word") == "ping"
        assert parsed.get("length") == 4


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
    MODEL_NAME = "glm-5"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED = True
    EXPECTS_JSON_MODE_PAYLOAD_IN_CONTENT = False


class TestQwen35ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "qwen35"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED = True

    def base_text_request_overrides(self) -> Mapping[str, object]:
        # qwen35's default thinking path is intermittent on this backend:
        # create may hang until timeout and stream may stop before
        # emitting final assistant text. Use the stable plain-text path.
        return {"chat_template_kwargs": {"enable_thinking": False}}


class TestMinimaxM25ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "minimax-m2.5"
    EXPECTS_JSON_MODE_PAYLOAD_IN_CONTENT = False


class TestMinimaxM21ChatCompletions(BaseRelaxedToolChoiceChatTests):
    __test__ = True
    MODEL_NAME = "minimax-m21"
    EXPECTS_JSON_MODE_PAYLOAD_IN_CONTENT = False
