from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODULE_PATH = REPO_ROOT / "multi_turn" / "probe_stream_behavior.py"
SPEC = importlib.util.spec_from_file_location("probe_stream_behavior", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def build_stream_chunk(
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    received_at_ms: float | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "choices": [{"delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    if received_at_ms is not None:
        chunk["_received_at_ms"] = received_at_ms
    return chunk


def test_summarize_stream_chunks_handles_delta_reasoning_only() -> None:
    chunks = [
        build_stream_chunk(delta={"content": "", "reasoning": "think-1"}, received_at_ms=10.0),
        build_stream_chunk(
            delta={"reasoning": "think-2"},
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            received_at_ms=20.0,
        ),
    ]

    summary = probe.summarize_stream_chunks(chunks, saw_done=True)

    assert summary["has_non_empty_delta_content"] is False
    assert summary["has_non_empty_reasoning"] is True
    assert summary["reasoning_only_stream"] is True
    assert summary["content_never_appears"] is True
    assert summary["aggregated_reasoning"] == "think-1think-2"
    assert summary["aggregated_content"] == ""
    assert summary["first_field_order"] == "reasoning_only"


def test_summarize_stream_chunks_detects_reasoning_before_content() -> None:
    chunks = [
        build_stream_chunk(delta={"reasoning": "plan"}, received_at_ms=5.0),
        build_stream_chunk(delta={"content": "answer"}, received_at_ms=12.0),
    ]

    summary = probe.summarize_stream_chunks(chunks, saw_done=True)

    assert summary["reasoning_before_content"] is True
    assert summary["first_field_order"] == "reasoning_before_content"
    assert summary["first_reasoning_field"] == "delta.reasoning"
    assert summary["aggregated_content"] == "answer"


def test_summarize_stream_chunks_handles_reasoning_content_variant() -> None:
    chunks = [
        build_stream_chunk(delta={"reasoning_content": "r1"}, received_at_ms=1.0),
        build_stream_chunk(delta={"reasoning_content": "r2"}, received_at_ms=2.0),
    ]

    summary = probe.summarize_stream_chunks(chunks, saw_done=True)

    assert summary["has_non_empty_reasoning"] is True
    assert summary["aggregated_reasoning"] == ""
    assert summary["aggregated_reasoning_content"] == "r1r2"
    assert summary["aggregated_combined_reasoning"] == "r1r2"
    assert summary["first_reasoning_field"] == "delta.reasoning_content"


def test_summarize_stream_chunks_supports_delta_text_fallback() -> None:
    chunks = [
        build_stream_chunk(delta={"text": "hel"}, received_at_ms=3.0),
        build_stream_chunk(delta={"text": "lo"}, received_at_ms=4.0),
    ]

    summary = probe.summarize_stream_chunks(chunks, saw_done=True)

    assert summary["has_non_empty_delta_content"] is False
    assert summary["has_non_empty_content_channel"] is True
    assert summary["aggregated_delta_content"] == ""
    assert summary["aggregated_delta_text"] == "hello"
    assert summary["aggregated_content"] == "hello"


def test_summarize_stream_chunks_marks_terminal_usage_only() -> None:
    chunks = [
        build_stream_chunk(delta={"content": "part1"}, received_at_ms=5.0),
        build_stream_chunk(
            delta={"content": "part2"},
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            received_at_ms=7.0,
        ),
    ]

    summary = probe.summarize_stream_chunks(chunks, saw_done=True)

    assert summary["finish_reasons"] == ["stop"]
    assert summary["usage_chunk_indexes"] == [2]
    assert summary["usage_only_on_terminal_chunk"] is True


def test_compare_stream_and_nonstream_reports_content_mismatch() -> None:
    stream_summary = probe.summarize_stream_chunks(
        [
            build_stream_chunk(delta={"reasoning": "only-reasoning"}, received_at_ms=1.0),
            build_stream_chunk(finish_reason="stop", usage={"total_tokens": 3}, received_at_ms=2.0),
        ],
        saw_done=True,
    )
    nonstream_summary = probe.summarize_nonstream_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "final-answer",
                        "reasoning": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    comparison = probe.compare_stream_and_nonstream(stream_summary, nonstream_summary)

    assert comparison["content_match"] is False
    assert comparison["combined_reasoning_match"] is False
