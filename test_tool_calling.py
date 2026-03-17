from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_MAX_COMPLETION_TOKENS,
    FailureArtifactRecorder,
    collect_stream_text,
    collect_stream_tool_calls as collect_http_stream_tool_calls,
    request_json,
    request_sse,
)


REPO_ROOT = Path(__file__).resolve().parent
README_PATH = str(REPO_ROOT / "README.md")


def build_weather_tool_definition() -> list[dict[str, object]]:
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


def build_list_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list",
                "description": "List files in a directory tree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "ignore": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_read_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file from disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["filePath"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_grep_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents with a regex pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "include": {"type": "string"},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_bash_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "workdir": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command", "description"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_edit_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit an existing file by replacing text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                        "replaceAll": {"type": "boolean"},
                    },
                    "required": ["filePath", "oldString", "newString"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_write_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write file contents to disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filePath", "content"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_task_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "task",
                "description": "Run a subtask with a subagent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": {"type": "string"},
                        "task_id": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "required": ["description", "prompt", "subagent_type"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_todowrite_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "todowrite",
                "description": "Update the todo list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                    "priority": {"type": "string"},
                                },
                                "required": ["content", "status", "priority"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["todos"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_sdk_weather_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def build_sdk_bash_and_grep_tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command", "description"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for patterns in files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "include": {"type": "string"},
                    },
                    "required": ["pattern", "path"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_process_data_tool_definition() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "process_data",
                "description": "Process a large data payload.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                    },
                    "required": ["data"],
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


def extract_tool_call_from_content(message: Mapping[str, object]) -> dict[str, object]:
    content = message.get("content")
    assert isinstance(content, str) and content.strip()

    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    assert parsed.get("name")

    arguments = parsed.get("arguments")
    if isinstance(arguments, dict):
        arguments_text = json.dumps(arguments)
    else:
        assert isinstance(arguments, str)
        arguments_text = arguments

    return {
        "type": "function",
        "function": {
            "name": parsed["name"],
            "arguments": arguments_text,
        },
    }


def extract_message_tool_calls(message: object) -> list[dict[str, object]]:
    tool_calls = getattr(message, "tool_calls", None) or []
    extracted: list[dict[str, object]] = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        extracted.append(
            {
                "id": getattr(tool_call, "id", ""),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(function, "name", ""),
                    "arguments": getattr(function, "arguments", ""),
                },
            }
        )
    return extracted


def ensure_tool_call_ids(tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {})
        assert isinstance(function, Mapping)
        normalized.append(
            {
                "id": tool_call.get("id") or f"stream-call-{index}",
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", ""),
                },
            }
        )
    return normalized


class BaseHTTPXToolCallingTests:
    __test__ = False

    MODEL_NAME: str = ""
    TOOL_REQUEST_MODE = "forced_named_tool_choice"
    SUPPORTS_EDIT_TOOL = True
    SUPPORTS_REPEATED_SAME_TOOL_CALL = True
    SUPPORTS_STREAM_TOOL_CALL = True
    SUPPORTS_STREAM_TOOL_ROUND_TRIP = True
    SUPPORTS_TASK_TOOL = True

    def tool_request_overrides(self) -> Mapping[str, object]:
        return {}

    def apply_tool_choice(self, payload: dict[str, object], tool_name: str) -> None:
        if self.TOOL_REQUEST_MODE == "forced_named_tool_choice":
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
        elif self.TOOL_REQUEST_MODE == "auto_tool_choice":
            payload["tool_choice"] = "auto"

    def build_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_weather_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the provided function to submit these exact arguments: "
                        "city='Tokyo', unit='celsius'. Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "collect_weather_args")
        payload.update(self.tool_request_overrides())
        return payload

    def build_tool_round_trip_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_weather_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the provided function to submit these exact arguments: "
                        "city='Tokyo', unit='celsius'. After the tool result is available, "
                        "reply with one short sentence summarizing it."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "collect_weather_args")
        payload.update(self.tool_request_overrides())
        return payload

    def build_stream_tool_payload(self) -> dict[str, object]:
        payload = self.build_tool_payload()
        payload["stream"] = True
        return payload

    def build_stream_tool_round_trip_payload(self) -> dict[str, object]:
        payload = self.build_tool_round_trip_payload()
        payload["stream"] = True
        return payload

    def build_repeated_same_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_weather_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Your next response must contain exactly two tool calls, both named "
                        "collect_weather_args, in the same assistant message. "
                        "The first tool call must use city='Tokyo', unit='celsius'. "
                        "The second tool call must use city='Shanghai', unit='celsius'. "
                        "Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "collect_weather_args")
        payload.update(self.tool_request_overrides())
        return payload

    def build_list_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_list_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the list tool exactly once with these exact arguments: "
                        f"path='{REPO_ROOT}', ignore=['.git/*']. "
                        "Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "list")
        payload.update(self.tool_request_overrides())
        return payload

    def build_read_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_read_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Your next response must be exactly one tool call named read. "
                        "Do not include any text before or after it. "
                        "Use the read tool exactly once with these exact arguments: "
                        f"filePath='{README_PATH}', offset=1, limit=20. "
                        "If you do not call the tool, the answer is invalid."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "read")
        payload.update(self.tool_request_overrides())
        return payload

    def build_grep_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_grep_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the grep tool exactly once with these exact arguments: "
                        f"pattern='test_', path='{REPO_ROOT}', include='*.py'. "
                        "Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "grep")
        payload.update(self.tool_request_overrides())
        return payload

    def build_bash_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_bash_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Use the bash tool exactly once with these exact arguments: "
                        f"command='pwd', workdir='{REPO_ROOT}', "
                        "description='Print current working directory'. "
                        "Do not answer with plain text."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "bash")
        payload.update(self.tool_request_overrides())
        return payload

    def build_edit_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_edit_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Your next response must be exactly one tool call named edit. "
                        "Do not include any text before or after it. "
                        "Use the edit tool exactly once with these exact arguments: "
                        f"filePath='{README_PATH}', oldString='# maas-test', "
                        "newString='# maas-test\\n', replaceAll=false. "
                        "If you do not call the tool, the answer is invalid."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "edit")
        payload.update(self.tool_request_overrides())
        return payload

    def build_write_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_write_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Your next response must be exactly one tool call named write. "
                        "Do not include any text before or after it. "
                        "Use the write tool exactly once with these exact arguments: "
                        f"filePath='{REPO_ROOT / 'opencode-write-demo.txt'}', "
                        "content='temporary write payload\\n'. "
                        "If you do not call the tool, the answer is invalid."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "write")
        payload.update(self.tool_request_overrides())
        return payload

    def build_task_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_task_tool_definition(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. When the task tool is available, you must use it.",
                },
                {
                    "role": "user",
                    "content": (
                        "Your next response must be exactly one tool call named task. "
                        "Do not include any text before or after it. "
                        "Use the task tool exactly once with these exact arguments: "
                        "description='inspect repo', prompt='Summarize the repo root files.', "
                        "subagent_type='worker'. If you do not call the tool, the answer is invalid."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "task")
        payload.update(self.tool_request_overrides())
        return payload

    def build_todowrite_tool_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.MODEL_NAME,
            "temperature": 0,
            "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
            "tools": build_todowrite_tool_definition(),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Your next response must be exactly one tool call named todowrite. "
                        "Do not include any text before or after it. "
                        "Use the todowrite tool exactly once with this exact todos array: "
                        "[{content:'Inspect API schema', status:'completed', priority:'high'}, "
                        "{content:'Add OpenCode tool tests', status:'in_progress', priority:'medium'}]. "
                        "If you do not call the tool, the answer is invalid."
                    ),
                },
            ],
        }
        self.apply_tool_choice(payload, "todowrite")
        payload.update(self.tool_request_overrides())
        return payload

    def build_tool_result_content(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "city": "Tokyo",
                "unit": "celsius",
                "summary": "Tokyo is 22C and clear.",
            }
        )

    def build_read_tool_result_content(self) -> str:
        return "\n".join(
            [
                f"<path>{README_PATH}</path>",
                "<type>file</type>",
                "<content>",
                "1: # maas-test",
                "2: 这是一个测试用的 README 片段。",
                "",
                "(End of file - total 2 lines)",
                "</content>",
            ]
        )

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
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "collect_weather_args"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["city"].lower().startswith("tokyo")
        assert arguments["unit"] == "celsius"

    def test_stream_returns_tool_call(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_stream_tool_payload(),
            recorder=failure_artifact_recorder,
        )
        stream_result = collect_http_stream_tool_calls(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert stream_result.chunk_count > 0
        assert stream_result.saw_done
        assert stream_result.tool_calls

        first = stream_result.tool_calls[0]
        assert first["type"] == "function"
        assert first["function"]["name"] == "collect_weather_args"

        arguments = parse_tool_arguments(first)
        assert arguments["city"].lower().startswith("tokyo")
        assert arguments["unit"] == "celsius"

    def test_tool_call_round_trip_returns_final_assistant_message(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        initial_payload = self.build_tool_round_trip_payload()
        initial_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            initial_payload,
            recorder=failure_artifact_recorder,
        )

        assistant_message = initial_completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(assistant_message)
        tool_result = self.build_tool_result_content()

        follow_up_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            {
                "model": self.MODEL_NAME,
                "temperature": 0,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                "messages": [
                    *initial_payload["messages"],
                    assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "collect_weather_args",
                        "content": tool_result,
                    },
                ],
            },
            recorder=failure_artifact_recorder,
        )

        follow_up_message = follow_up_completion["choices"][0]["message"]
        assert follow_up_message["role"] == "assistant"
        assert isinstance(follow_up_message["content"], str)
        assert follow_up_message["content"].strip()
        text = follow_up_message["content"].lower()
        assert "tokyo" in text or "celsius" in text

    def test_stream_tool_call_round_trip_returns_final_assistant_message(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        initial_payload = self.build_stream_tool_round_trip_payload()
        response, events, raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            initial_payload,
            recorder=failure_artifact_recorder,
        )
        initial_stream_result = collect_http_stream_tool_calls(events)

        assert response.status_code == 200, raw_text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert initial_stream_result.chunk_count > 0
        assert initial_stream_result.saw_done
        assert initial_stream_result.tool_calls

        streamed_tool_calls = ensure_tool_call_ids(initial_stream_result.tool_calls)
        tool_call = streamed_tool_calls[0]
        tool_result = self.build_tool_result_content()

        follow_up_response, follow_up_events, follow_up_raw_text = request_sse(
            http_client,
            CHAT_COMPLETIONS_PATH,
            {
                "model": self.MODEL_NAME,
                "temperature": 0,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                "stream": True,
                "messages": [
                    *initial_payload["messages"],
                    {
                        "role": "assistant",
                        "tool_calls": streamed_tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "collect_weather_args",
                        "content": tool_result,
                    },
                ],
            },
            recorder=failure_artifact_recorder,
        )
        follow_up_stream_result = collect_stream_text(follow_up_events)

        assert follow_up_response.status_code == 200, follow_up_raw_text
        assert follow_up_response.headers["content-type"].startswith("text/event-stream")
        assert follow_up_stream_result.chunk_count > 0
        assert follow_up_stream_result.saw_done
        assert follow_up_stream_result.text
        text = follow_up_stream_result.text.lower()
        assert "tokyo" in text or "celsius" in text

    def test_create_returns_repeated_same_tool_calls(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_repeated_same_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        assert isinstance(tool_calls, list)
        assert len(tool_calls) >= 2

        first = tool_calls[0]
        second = tool_calls[1]
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        assert first["type"] == "function"
        assert second["type"] == "function"
        assert first["function"]["name"] == "collect_weather_args"
        assert second["function"]["name"] == "collect_weather_args"

        first_arguments = parse_tool_arguments(first)
        second_arguments = parse_tool_arguments(second)
        assert first_arguments["city"].lower().startswith("tokyo")
        assert first_arguments["unit"] == "celsius"
        assert second_arguments["city"].lower().startswith("shanghai")
        assert second_arguments["unit"] == "celsius"

    def test_list_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_list_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "list"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["path"] == str(REPO_ROOT)
        assert isinstance(arguments["ignore"], list)
        assert ".git/*" in arguments["ignore"]

    def test_read_tool_round_trip_returns_final_assistant_message(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        initial_payload = self.build_read_tool_payload()
        initial_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            initial_payload,
            recorder=failure_artifact_recorder,
        )

        assistant_message = initial_completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(assistant_message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "read"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["filePath"] == README_PATH
        assert arguments["offset"] == 1
        assert arguments["limit"] == 20

        follow_up_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            {
                "model": self.MODEL_NAME,
                "temperature": 0,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                "messages": [
                    *initial_payload["messages"],
                    assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "read",
                        "content": self.build_read_tool_result_content(),
                    },
                ],
            },
            recorder=failure_artifact_recorder,
        )

        follow_up_message = follow_up_completion["choices"][0]["message"]
        assert follow_up_message["role"] == "assistant"
        assert isinstance(follow_up_message["content"], str)
        assert follow_up_message["content"].strip()

    def test_grep_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_grep_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "grep"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["pattern"] == "test_"
        assert arguments["path"] == str(REPO_ROOT)
        assert arguments["include"] == "*.py"

    def test_bash_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_bash_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "bash"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["command"] == "pwd"
        assert isinstance(arguments["description"], str)
        assert "working directory" in arguments["description"].lower()
        workdir = arguments.get("workdir")
        if workdir is not None:
            assert workdir == str(REPO_ROOT)
        timeout = arguments.get("timeout")
        if timeout is not None:
            assert isinstance(timeout, int)

    def test_edit_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_edit_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "edit"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["filePath"] == README_PATH
        assert arguments["oldString"] == "# maas-test"
        assert arguments["newString"].rstrip("\n") == "# maas-test"
        replace_all = arguments.get("replaceAll")
        if replace_all is not None:
            assert replace_all is False

    def test_write_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_write_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "write"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["filePath"] == str(REPO_ROOT / "opencode-write-demo.txt")
        assert arguments["content"].rstrip("\n") == "temporary write payload"

    def test_task_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_task_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "task"

        arguments = parse_tool_arguments(tool_call)
        assert arguments["description"] == "inspect repo"
        assert arguments["prompt"] == "Summarize the repo root files."
        assert arguments["subagent_type"] == "worker"

    def test_todowrite_tool_returns_valid_arguments(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            self.build_todowrite_tool_payload(),
            recorder=failure_artifact_recorder,
        )

        message = completion["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_call = extract_first_tool_call(message)
        else:
            tool_call = extract_tool_call_from_content(message)
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "todowrite"

        arguments = parse_tool_arguments(tool_call)
        todos = arguments["todos"]
        assert isinstance(todos, list)
        assert len(todos) == 2
        assert todos[0]["content"] == "Inspect API schema"
        assert todos[0]["status"] == "completed"
        assert todos[0]["priority"] == "high"
        assert todos[1]["content"] == "Add OpenCode tool tests"
        assert todos[1]["status"] == "in_progress"
        assert todos[1]["priority"] == "medium"

    def test_create_accepts_multi_turn_history_with_assistant_and_tool_messages(
        self,
        http_client: httpx.Client,
        failure_artifact_recorder: FailureArtifactRecorder,
    ) -> None:
        initial_payload = self.build_tool_payload()
        initial_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            initial_payload,
            recorder=failure_artifact_recorder,
        )

        assistant_message = initial_completion["choices"][0]["message"]
        tool_call = extract_first_tool_call(assistant_message)
        history_completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            {
                "model": self.MODEL_NAME,
                "temperature": 0,
                "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
                "messages": [
                    *initial_payload["messages"],
                    assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "collect_weather_args",
                        "content": self.build_tool_result_content(),
                    },
                    {
                        "role": "user",
                        "content": "Based only on the tool result above, reply with 'Tokyo celsius'.",
                    },
                ],
            },
            recorder=failure_artifact_recorder,
        )

        history_message = history_completion["choices"][0]["message"]
        assert history_message["role"] == "assistant"
        assert isinstance(history_message["content"], str)
        assert history_message["content"].strip()
        text = history_message["content"].lower()
        assert "tokyo" in text
        assert "celsius" in text

class BaseOpenAICompatibleToolCallingTests(BaseHTTPXToolCallingTests):
    __test__ = False


class BaseRelaxedToolChoiceToolCallingTests(BaseHTTPXToolCallingTests):
    __test__ = False

    TOOL_REQUEST_MODE = "auto_tool_choice"


class TestKimiK25ToolCalling(BaseOpenAICompatibleToolCallingTests):
    __test__ = True
    MODEL_NAME = "kimi-k25"
    SUPPORTS_REPEATED_SAME_TOOL_CALL = False


class TestGLM5ToolCalling(BaseRelaxedToolChoiceToolCallingTests):
    __test__ = True
    MODEL_NAME = "glm5"


class TestQwen35ToolCalling(BaseRelaxedToolChoiceToolCallingTests):
    __test__ = True
    MODEL_NAME = "qwen35"
    SUPPORTS_EDIT_TOOL = False
    SUPPORTS_REPEATED_SAME_TOOL_CALL = False
    SUPPORTS_STREAM_TOOL_ROUND_TRIP = False
    SUPPORTS_TASK_TOOL = False


class TestMinimaxM25ToolCalling(BaseRelaxedToolChoiceToolCallingTests):
    __test__ = True
    MODEL_NAME = "minimax-m25"


class TestMinimaxM21ToolCalling(BaseRelaxedToolChoiceToolCallingTests):
    __test__ = True
    MODEL_NAME = "minimax-m21"
    SUPPORTS_REPEATED_SAME_TOOL_CALL = False


def collect_stream_tool_calls(
    sdk_client: OpenAI,
    request_payload: dict[str, object],
    failure_artifact_recorder: FailureArtifactRecorder,
) -> tuple[list[dict[str, object]], int]:
    try:
        stream = sdk_client.chat.completions.create(**request_payload)
    except Exception as exc:
        failure_artifact_recorder.add_sdk_exchange(
            api="chat.completions.create",
            request_payload=request_payload,
            exception=exc,
        )
        raise

    stream_chunks: list[dict[str, object]] = []
    collected: dict[int, dict[str, object]] = {}
    chunk_count = 0

    try:
        for chunk in stream:
            chunk_count += 1
            stream_chunks.append(chunk.model_dump())

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for partial in delta_tool_calls:
                index = getattr(partial, "index", 0) or 0
                current = collected.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    },
                )

                partial_id = getattr(partial, "id", None)
                if partial_id:
                    current["id"] = partial_id

                partial_type = getattr(partial, "type", None)
                if partial_type:
                    current["type"] = partial_type

                function = getattr(partial, "function", None)
                if function is None:
                    continue

                name = getattr(function, "name", None)
                if name:
                    current["function"]["name"] = name

                arguments = getattr(function, "arguments", None)
                if arguments:
                    current["function"]["arguments"] += arguments
    except Exception as exc:
        failure_artifact_recorder.add_sdk_exchange(
            api="chat.completions.create",
            request_payload=request_payload,
            stream_chunks=stream_chunks,
            exception=exc,
        )
        raise

    failure_artifact_recorder.add_sdk_exchange(
        api="chat.completions.create",
        request_payload=request_payload,
        stream_chunks=stream_chunks,
    )

    ordered = [collected[index] for index in sorted(collected)]
    return ordered, chunk_count


def request_completion(
    sdk_client: OpenAI,
    request_payload: dict[str, object],
    failure_artifact_recorder: FailureArtifactRecorder,
):
    try:
        completion = sdk_client.chat.completions.create(**request_payload)
    except Exception as exc:
        failure_artifact_recorder.add_sdk_exchange(
            api="chat.completions.create",
            request_payload=request_payload,
            exception=exc,
        )
        raise

    failure_artifact_recorder.add_sdk_exchange(
        api="chat.completions.create",
        request_payload=request_payload,
        response=completion,
    )
    return completion


@pytest.mark.tool_calling_probe
def test_sdk_single_tool_call_returns_valid_json_arguments(
    sdk_client: OpenAI,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Use the get_weather tool exactly once with location='Beijing' "
                    "and unit='celsius'. Do not answer with plain text."
                ),
            },
        ],
        "tools": build_sdk_weather_tool_definition(),
        "tool_choice": "auto",
    }

    completion = request_completion(sdk_client, request_payload, failure_artifact_recorder)
    message = completion.choices[0].message
    tool_calls = extract_message_tool_calls(message)

    assert tool_calls
    first = tool_calls[0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_weather"

    arguments = json.loads(first["function"]["arguments"])
    assert arguments["location"].lower().startswith("beijing")
    assert arguments["unit"] == "celsius"


@pytest.mark.tool_calling_probe
def test_sdk_stream_tool_call_emits_valid_json_arguments(
    sdk_client: OpenAI,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Your next response must be exactly one tool call named get_weather. "
                    "Use location='Beijing' and unit='celsius'."
                ),
            },
        ],
        "tools": build_sdk_weather_tool_definition(),
        "tool_choice": "auto",
    }

    tool_calls, chunk_count = collect_stream_tool_calls(sdk_client, request_payload, failure_artifact_recorder)

    assert chunk_count > 0
    assert tool_calls
    first = tool_calls[0]
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_weather"

    arguments = json.loads(first["function"]["arguments"])
    assert arguments["location"].lower().startswith("beijing")
    assert arguments["unit"] == "celsius"


@pytest.mark.tool_calling_probe
def test_sdk_multi_tool_request_returns_valid_tool_calls(
    sdk_client: OpenAI,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Use one or more of the provided tools before answering. "
                    f"You may inspect the repo at '{REPO_ROOT}' and print the working directory. "
                    "Do not answer with plain text."
                ),
            },
        ],
        "tools": build_sdk_bash_and_grep_tool_definitions(),
        "tool_choice": "auto",
    }

    completion = request_completion(sdk_client, request_payload, failure_artifact_recorder)
    message = completion.choices[0].message
    tool_calls = extract_message_tool_calls(message)

    assert tool_calls
    for tool_call in tool_calls:
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] in {"bash", "grep"}
        arguments = json.loads(tool_call["function"]["arguments"])
        assert isinstance(arguments, Mapping)


@pytest.mark.tool_calling_probe
def test_sdk_large_tool_arguments_remain_valid_json(
    sdk_client: OpenAI,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": "You are a data processor."},
            {
                "role": "user",
                "content": (
                    "Use the process_data tool exactly once. "
                    "Set data to a comma-separated string containing the numbers 1 through 400. "
                    "Do not answer with plain text."
                ),
            },
        ],
        "tools": build_process_data_tool_definition(),
        "tool_choice": "auto",
    }

    completion = request_completion(sdk_client, request_payload, failure_artifact_recorder)
    message = completion.choices[0].message
    tool_calls = extract_message_tool_calls(message)

    assert tool_calls
    first = tool_calls[0]
    assert first["function"]["name"] == "process_data"

    arguments_text = first["function"]["arguments"]
    arguments = json.loads(arguments_text)
    data = arguments["data"]
    assert isinstance(data, str)
    assert len(arguments_text) > 256
    assert data.startswith("1,2,3,4,5")
