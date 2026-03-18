from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    FailureArtifactRecorder,
    get_api_key,
    request_json,
    resolve_base_url,
)
from k2_verifier.core import DatasetCase, ToolCallsValidator, load_dataset_cases


TESTS_DIR = Path(__file__).resolve().parent
DATASET_PATH = TESTS_DIR / "fixtures" / "k2" / "tool_calling_subset.jsonl"
DATASET_CASES = load_dataset_cases(DATASET_PATH)
B7_STABLE_SKIP_MODELS = {"qwen35", "kimi-k25"}
B7_EXPECTED_TOOL_SEQUENCE = [
    "fetch_seed_word",
    "uppercase_word",
    "decorate_word",
]
B7_CHAIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_seed_word",
            "description": "Return a seed word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "uppercase_word",
            "description": "Uppercase a word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                },
                "required": ["word"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decorate_word",
            "description": "Add a prefix and suffix to a word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "prefix": {"type": "string"},
                    "suffix": {"type": "string"},
                },
                "required": ["word", "prefix", "suffix"],
                "additionalProperties": False,
            },
        },
    },
]


def _build_b7_payload(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 512,
        "tools": B7_CHAIN_TOOLS,
        "tool_choice": "auto",
        "messages": messages,
    }


def _initial_b7_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Follow tool instructions exactly."},
        {
            "role": "user",
            "content": (
                "Use tools only. Call exactly one tool per assistant turn. "
                "Step 1: call fetch_seed_word with topic='chain-demo'. "
                "Step 2: after that tool result, call uppercase_word with the returned word. "
                "Step 3: after that tool result, call decorate_word with the returned uppercase word, "
                "prefix='[', suffix=']'. "
                "Step 4: after that tool result, reply with exactly [STONE] and nothing else. "
                "Do not skip or merge steps."
            ),
        },
    ]


def _normalize_tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    tool_call_id = tool_call.get("id")
    if isinstance(tool_call_id, str) and tool_call_id:
        return tool_call_id

    function_name = str(tool_call.get("function", {}).get("name") or "tool")
    normalized = f"functions.{function_name}:{index}"
    tool_call["id"] = normalized
    return normalized


def _execute_b7_tool_call(tool_call: dict[str, Any]) -> tuple[str, str]:
    function = tool_call.get("function") or {}
    tool_name = str(function.get("name") or "")
    arguments = json.loads(str(function.get("arguments") or "{}"))

    if tool_name == "fetch_seed_word":
        assert arguments == {"topic": "chain-demo"}
        return tool_name, json.dumps({"word": "stone"})

    if tool_name == "uppercase_word":
        assert arguments == {"word": "stone"}
        return tool_name, json.dumps({"word": "STONE"})

    if tool_name == "decorate_word":
        assert arguments == {"word": "STONE", "prefix": "[", "suffix": "]"}
        return tool_name, json.dumps({"decorated_word": "[STONE]"})

    raise AssertionError(f"Unexpected tool name for B7 chain: {tool_name!r}")


def _run_b7_chain(
    *,
    model: str,
    http_client,
    recorder: FailureArtifactRecorder,
) -> tuple[list[str], str]:
    messages = _initial_b7_messages()
    executed_tool_names: list[str] = []

    for _ in range(len(B7_EXPECTED_TOOL_SEQUENCE) + 1):
        completion = request_json(
            http_client,
            CHAT_COMPLETIONS_PATH,
            _build_b7_payload(model, messages),
            recorder=recorder,
        )
        message = completion["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            assert isinstance(tool_calls, list)
            assert len(tool_calls) == 1, completion
            tool_call = tool_calls[0]
            assert isinstance(tool_call, dict), completion

            tool_name, tool_result = _execute_b7_tool_call(tool_call)
            executed_tool_names.append(tool_name)
            tool_call_id = _normalize_tool_call_id(tool_call, len(executed_tool_names) - 1)

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )
            continue

        content = str(message.get("content") or "").strip()
        assert content, completion
        return executed_tool_names, content

    raise AssertionError(f"B7 tool chain did not finish within the expected number of rounds for {model}")


async def _run_dataset_case(
    *,
    case: DatasetCase,
    model: str,
    recorder: FailureArtifactRecorder,
    tmp_path: Path,
) -> dict[str, object]:
    output_path = tmp_path / f"{case.case_id}-results.jsonl"
    summary_path = tmp_path / f"{case.case_id}-summary.json"

    async with ToolCallsValidator(
        model=model,
        base_url=resolve_base_url().rstrip("/"),
        api_key=get_api_key(),
        output_file=output_path,
        summary_file=summary_path,
    ) as validator:
        request_record = validator.build_request_record(case.to_dataset_entry(), data_index=case.data_index)
        api_name = "completions.create" if validator.use_raw_completions else "chat.completions.create"

        try:
            result = await validator.process_request(request_record, case.data_index)
        except Exception as exc:
            recorder.add_sdk_exchange(
                api=api_name,
                request_payload=request_record["prepared"],
                exception=exc,
            )
            raise

        recorder.add_sdk_exchange(
            api=api_name,
            request_payload=request_record["prepared"],
            response=result["response"],
        )
        return result


@pytest.mark.parametrize("case", DATASET_CASES, ids=[case.pytest_id for case in DATASET_CASES])
def test_dataset_driven_tool_calling_case(
    model: str,
    case: DatasetCase,
    tmp_path: Path,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    if case.should_skip_model(model):
        pytest.skip(f"{case.case_id} is not in the stable passing path for {model}")

    result = asyncio.run(
        _run_dataset_case(
            case=case,
            model=model,
            recorder=failure_artifact_recorder,
            tmp_path=tmp_path,
        )
    )

    response = result["response"]
    assert result["status"] == "success", response

    choices = response.get("choices")
    assert isinstance(choices, list) and choices, response

    if case.expected_finish_reason == "tool_calls":
        assert result["tool_calls_present"] is True, response
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls")
        assert isinstance(tool_calls, list) and tool_calls, response
    else:
        assert result["finish_reason"] == case.expected_finish_reason, response

    if case.expected_tool_calls_valid is not None:
        assert result["tool_calls_valid"] is case.expected_tool_calls_valid, response

    if case.expected_tool_call_names:
        assert result["tool_call_names_match"] is True, response


def test_multi_step_tool_chain_round_trip(
    model: str,
    http_client,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    if model in B7_STABLE_SKIP_MODELS:
        pytest.skip(f"B7 multi-step tool chain is not in the stable passing path for {model}")

    executed_tool_names, final_text = _run_b7_chain(
        model=model,
        http_client=http_client,
        recorder=failure_artifact_recorder,
    )

    assert executed_tool_names == B7_EXPECTED_TOOL_SEQUENCE
    assert final_text.strip().strip("`").strip() == "[STONE]"
