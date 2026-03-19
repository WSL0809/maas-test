from __future__ import annotations

import asyncio
import json
from pathlib import Path

from k2_verifier.core import (
    DatasetCase,
    StreamAccumulator,
    ToolCallsValidator,
    build_default_report_paths,
    build_summary,
    load_dataset_cases,
    merge_delta_tool_calls,
    parse_extra_body,
    prepare_request_payload,
    validate_tool_call_against_tools,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_DIR = REPO_ROOT / "tests"
FIXTURE_PATH = TESTS_DIR / "fixtures" / "k2" / "verifier_smoke.jsonl"
SUBSET_FIXTURE_PATH = TESTS_DIR / "fixtures" / "k2" / "tool_calling_subset.jsonl"


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
def test_prepare_request_payload_applies_overrides_and_normalizes_tool_history() -> None:
    raw_request = {
        "messages": [
            {"role": "_input", "content": "System prompt."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "serach:0",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "serach:0", "content": "{}"},
            {"role": "user", "content": "Reply with quartz."},
        ],
        "stream": True,
        "max_completion_tokens": 32,
    }

    prepared = prepare_request_payload(
        raw_request,
        model="kimi-k2",
        temperature=0.6,
        max_tokens=128,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        normalize_tool_call_ids=True,
    )

    assert prepared["model"] == "kimi-k2"
    assert prepared["temperature"] == 0.6
    assert prepared["max_completion_tokens"] == 128
    assert prepared["messages"][0]["role"] == "system"
    assert prepared["messages"][1]["tool_calls"][0]["id"] == "functions.search:0"
    assert prepared["messages"][2]["tool_call_id"] == "functions.search:0"
    assert prepared["stream_options"]["include_usage"] is True
    assert prepared["_extra_body_preview"] == {"chat_template_kwargs": {"thinking": False}}


def test_prepare_request_payload_converts_raw_completions_with_tokenizer() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, tools, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            assert tools == build_weather_tool_definition()
            return f"prompt:{messages[-1]['content']}"

    prepared = prepare_request_payload(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Call the function."},
            ],
            "tools": build_weather_tool_definition(),
            "max_completion_tokens": 64,
        },
        model="kimi-k2",
        max_tokens=128,
        use_raw_completions=True,
        tokenizer=FakeTokenizer(),
    )

    assert prepared["prompt"] == "prompt:Call the function."
    assert "messages" not in prepared
    assert "tools" not in prepared
    assert prepared["max_tokens"] == 128


def test_stream_accumulator_reassembles_content_reasoning_and_tool_calls() -> None:
    accumulator = StreamAccumulator()
    accumulator.add_chat_delta(
        content="Hel",
        reasoning="think-1",
        delta_tool_calls=[
            {
                "index": 0,
                "id": "call_0",
                "type": "function",
                "function": {"name": "collect_weather_args", "arguments": '{"city":"Tok'},
            }
        ],
    )
    accumulator.add_chat_delta(
        content="lo",
        reasoning="think-2",
        delta_tool_calls=[
            {
                "index": 0,
                "function": {"arguments": 'yo","unit":"celsius"}'},
            }
        ],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )

    response = accumulator.build_response(
        request={"model": "kimi-k2"},
        request_id="req_1",
        created=123,
        use_raw_completions=False,
    )

    choice = response["choices"][0]
    assert choice["message"]["content"] == "Hello"
    assert choice["message"]["reasoning_content"] == "think-1think-2"
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city":"Tokyo","unit":"celsius"}'
    assert response["usage"]["total_tokens"] == 14


def test_merge_delta_tool_calls_accepts_mapping_fragments() -> None:
    collected: dict[int, dict[str, object]] = {}
    merge_delta_tool_calls(
        collected,
        [
            {
                "index": 1,
                "id": "call_1",
                "type": "function",
                "function": {"name": "collect_weather_args", "arguments": '{"city":"Shang'},
            },
            {
                "index": 1,
                "function": {"arguments": 'hai","unit":"celsius"}'},
            },
        ],
    )

    assert collected[1]["id"] == "call_1"
    assert collected[1]["function"]["name"] == "collect_weather_args"
    assert collected[1]["function"]["arguments"] == '{"city":"Shanghai","unit":"celsius"}'


def test_validate_tool_call_against_tools_checks_schema() -> None:
    tools = build_weather_tool_definition()

    valid_tool_call = {
        "type": "function",
        "function": {
            "name": "collect_weather_args",
            "arguments": json.dumps({"city": "Tokyo", "unit": "celsius"}),
        },
    }
    invalid_tool_call = {
        "type": "function",
        "function": {
            "name": "collect_weather_args",
            "arguments": json.dumps({"city": "Tokyo", "unit": "kelvin"}),
        },
    }

    assert validate_tool_call_against_tools(valid_tool_call, tools) is True
    assert validate_tool_call_against_tools(invalid_tool_call, tools) is False


def test_build_summary_counts_stop_tool_calls_and_failures() -> None:
    summary = build_summary(
        [
            {
                "status": "success",
                "finish_reason": "stop",
                "tool_calls_valid": None,
                "response": {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            },
            {
                "status": "success",
                "finish_reason": "tool_calls",
                "tool_calls_valid": True,
                "response": {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
            },
            {
                "status": "failed",
                "finish_reason": "length",
                "tool_calls_valid": False,
                "response": {},
            },
        ],
        model="kimi-k2",
        eval_started_at="2026-03-18T12:00:00",
        eval_finished_at="2026-03-18T12:01:00",
        eval_duration_ms=60000,
    )

    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["finish_stop"] == 1
    assert summary["finish_tool_calls"] == 1
    assert summary["finish_others"] == 1
    assert summary["finish_others_detail"] == {"length": 1}
    assert summary["successful_tool_call_count"] == 1
    assert summary["schema_validation_error_count"] == 0
    assert summary["usage"]["total_tokens"] == 11


def test_load_dataset_cases_reads_metadata_and_skip_models() -> None:
    cases = load_dataset_cases(SUBSET_FIXTURE_PATH)

    assert cases
    first = cases[0]
    assert isinstance(first, DatasetCase)
    assert first.case_id == "single_tool_nonstream"
    assert first.expected_finish_reason == "tool_calls"
    assert first.expected_tool_calls_valid is True
    assert first.expected_tool_call_names == ()
    assert first.request["messages"][0]["role"] == "system"

    repeated = next(case for case in cases if case.case_id == "repeated_same_tool_calls")
    assert repeated.should_skip_model("kimi-k25") is True
    assert repeated.should_skip_model("glm-5") is False
    assert repeated.expected_tool_call_names == (
        "collect_weather_args",
        "collect_weather_args",
    )

    parallel = next(case for case in cases if case.case_id == "parallel_distinct_tool_calls")
    assert parallel.expected_tool_call_names == (
        "lookup_weather",
        "lookup_local_time",
    )
    assert parallel.should_skip_model("qwen35") is True
    assert parallel.should_skip_model("minimax-m21") is True
    assert parallel.should_skip_model("glm-5") is False


def test_validate_request_processes_single_dataset_case(monkeypatch, tmp_path) -> None:
    async def fake_send_request(self, request):
        assert request["model"] == "kimi-k2"
        return (
            "success",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "collect_weather_args",
                                        "arguments": json.dumps(
                                            {"city": "Tokyo", "unit": "celsius"}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
            },
        )

    monkeypatch.setattr(ToolCallsValidator, "send_request", fake_send_request)
    case = load_dataset_cases(SUBSET_FIXTURE_PATH)[0]

    async def run_case() -> dict[str, object]:
        async with ToolCallsValidator(
            model="kimi-k2",
            base_url="https://example.test/v1",
            api_key="demo",
            output_file=tmp_path / "results.jsonl",
            summary_file=tmp_path / "summary.json",
        ) as validator:
            return await validator.validate_request(case.to_dataset_entry(), data_index=case.data_index)

    result = asyncio.run(run_case())

    assert result["case_id"] == "single_tool_nonstream"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls_present"] is True
    assert result["tool_calls_valid"] is True
    assert result["tool_call_names"] == ["collect_weather_args"]
    assert result["tool_call_names_match"] is None


def test_validate_request_marks_tool_calls_present_even_when_finish_reason_is_stop(monkeypatch, tmp_path) -> None:
    async def fake_send_request(self, request):
        assert request["model"] == "kimi-k2"
        return (
            "success",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "collect_weather_args",
                                        "arguments": json.dumps(
                                            {"city": "Tokyo", "unit": "celsius"}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
            },
        )

    monkeypatch.setattr(ToolCallsValidator, "send_request", fake_send_request)
    case = load_dataset_cases(SUBSET_FIXTURE_PATH)[0]

    async def run_case() -> dict[str, object]:
        async with ToolCallsValidator(
            model="kimi-k2",
            base_url="https://example.test/v1",
            api_key="demo",
            output_file=tmp_path / "results.jsonl",
            summary_file=tmp_path / "summary.json",
        ) as validator:
            return await validator.validate_request(case.to_dataset_entry(), data_index=case.data_index)

    result = asyncio.run(run_case())

    assert result["finish_reason"] == "stop"
    assert result["tool_calls_present"] is True
    assert result["tool_calls_valid"] is True
    assert result["tool_call_names_match"] is None


def test_validate_request_matches_expected_tool_call_names_ignoring_order(monkeypatch, tmp_path) -> None:
    async def fake_send_request(self, request):
        assert request["model"] == "kimi-k2"
        return (
            "success",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_local_time",
                                        "arguments": json.dumps({"city": "Tokyo"}),
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": json.dumps(
                                            {"city": "Tokyo", "unit": "celsius"}
                                        ),
                                    },
                                },
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    monkeypatch.setattr(ToolCallsValidator, "send_request", fake_send_request)
    case = next(
        dataset_case
        for dataset_case in load_dataset_cases(SUBSET_FIXTURE_PATH)
        if dataset_case.case_id == "parallel_distinct_tool_calls"
    )

    async def run_case() -> dict[str, object]:
        async with ToolCallsValidator(
            model="kimi-k2",
            base_url="https://example.test/v1",
            api_key="demo",
            output_file=tmp_path / "results.jsonl",
            summary_file=tmp_path / "summary.json",
        ) as validator:
            return await validator.validate_request(case.to_dataset_entry(), data_index=case.data_index)

    result = asyncio.run(run_case())

    assert result["tool_calls_valid"] is True
    assert result["tool_call_names"] == ["lookup_local_time", "lookup_weather"]
    assert result["expected_tool_call_names"] == ["lookup_weather", "lookup_local_time"]
    assert result["tool_call_names_match"] is True


def test_save_result_and_update_stats_flushes_summary_after_each_result(tmp_path) -> None:
    output_path, summary_path = build_default_report_paths(
        str(tmp_path / "results.jsonl"),
        str(tmp_path / "summary.json"),
    )

    partial_result = {
        "data_index": 1,
        "request": {"model": "kimi-k2"},
        "response": {"usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9}},
        "status": "success",
        "finish_reason": "tool_calls",
        "tool_calls_present": True,
        "tool_calls_valid": True,
        "last_run_at": "2026-03-18T15:00:00",
        "duration_ms": 123,
        "hash": "case-1",
    }

    async def run_save() -> None:
        async with ToolCallsValidator(
            model="kimi-k2",
            base_url="https://example.test/v1",
            api_key="demo",
            output_file=output_path,
            summary_file=summary_path,
        ) as validator:
            validator.results = [partial_result]
            validator.eval_started_at = "2026-03-18T15:00:00"
            validator.eval_start_ts = 1.0
            validator.eval_end_ts = 2.0
            validator.eval_finished_at = None
            await validator.save_result_and_update_stats(partial_result)

    asyncio.run(run_save())

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["success_count"] == 1
    assert summary["finish_tool_calls"] == 1
    assert summary["successful_tool_call_count"] == 1
    assert summary["usage"]["total_tokens"] == 9


def test_validate_file_writes_results_and_summary_from_synthetic_fixture(tmp_path, monkeypatch) -> None:
    output_path, summary_path = build_default_report_paths(
        str(tmp_path / "results.jsonl"),
        str(tmp_path / "summary.json"),
    )

    async def fake_send_request(self, request):
        last_message = request["messages"][-1]["content"]
        if "provided function" in last_message:
            return (
                "success",
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "collect_weather_args",
                                            "arguments": json.dumps(
                                                {"city": "Tokyo", "unit": "celsius"}
                                            ),
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
                },
            )
        return (
            "success",
            {
                "choices": [
                    {
                        "message": {"content": "quartz"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(ToolCallsValidator, "send_request", fake_send_request)

    async def run_validator() -> None:
        async with ToolCallsValidator(
            model="kimi-k2",
            base_url="https://example.test/v1",
            api_key="demo",
            output_file=output_path,
            summary_file=summary_path,
        ) as validator:
            await validator.validate_file(FIXTURE_PATH)

    asyncio.run(run_validator())

    result_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(result_rows) == 2
    assert result_rows[0]["data_index"] == 1
    assert result_rows[1]["data_index"] == 2
    assert result_rows[0]["finish_reason"] == "tool_calls"
    assert result_rows[0]["tool_calls_valid"] is True
    assert result_rows[1]["finish_reason"] == "stop"
    assert summary["success_count"] == 2
    assert summary["finish_tool_calls"] == 1
    assert summary["finish_stop"] == 1
    assert summary["successful_tool_call_count"] == 1
    assert summary["usage"]["total_tokens"] == 12


def test_parse_extra_body_requires_object_json() -> None:
    assert parse_extra_body('{"chat_template_kwargs":{"thinking":false}}') == {
        "chat_template_kwargs": {"thinking": False}
    }
