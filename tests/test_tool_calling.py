from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.chat_test_support import FailureArtifactRecorder, get_api_key, resolve_base_url
from k2_verifier.core import DatasetCase, ToolCallsValidator, load_dataset_cases


TESTS_DIR = Path(__file__).resolve().parent
DATASET_PATH = TESTS_DIR / "fixtures" / "k2" / "tool_calling_subset.jsonl"
DATASET_CASES = load_dataset_cases(DATASET_PATH)


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
