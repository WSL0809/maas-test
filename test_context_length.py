from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

from chat_test_support import (
    CHAT_COMPLETIONS_PATH,
    FailureArtifactRecorder,
    request_response,
)


MODELS_PATH = "models"
MAX_COMPLETION_TOKENS = 1
FILLER_UNIT = " x"
MAX_EXPONENTIAL_SEARCH_STEPS = 20
CONTEXT_OVERFLOW_MARKERS = (
    "max_model_len",
    "maximum context length",
    "context length",
    "prompt is too long",
    "requested",
    "too many tokens",
)


@dataclass(frozen=True)
class ContextProbeResult:
    repetitions: int
    status_code: int
    response_text: str
    outcome: Literal["success", "overflow"]


def build_context_messages(repetitions: int) -> list[dict[str, str]]:
    filler = FILLER_UNIT * repetitions
    return [
        {
            "role": "user",
            "content": (
                "This is a context-length probe. "
                "Reply with the single digit 1."
                f"{filler}"
            ),
        }
    ]


def build_context_payload(model: str, repetitions: int) -> dict[str, object]:
    return {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": build_context_messages(repetitions),
    }


def assert_model_is_listed(
    http_client: httpx.Client,
    model: str,
    recorder: FailureArtifactRecorder,
) -> None:
    response = request_response(http_client, "GET", MODELS_PATH, recorder=recorder)
    assert response.status_code == 200, response.text

    payload = response.json()
    data = payload.get("data")
    assert isinstance(data, list) and data, payload

    model_ids = [model_card.get("id") for model_card in data if isinstance(model_card, dict)]
    assert model in model_ids, f"/v1/models did not report model {model!r}: {response.text}"


def is_context_overflow_response(response: httpx.Response) -> bool:
    if response.status_code == 200:
        return False
    response_text = response.text.lower()
    return any(marker in response_text for marker in CONTEXT_OVERFLOW_MARKERS)


def probe_context_length(
    http_client: httpx.Client,
    model: str,
    repetitions: int,
    recorder: FailureArtifactRecorder,
) -> ContextProbeResult:
    payload = build_context_payload(model, repetitions)
    response = request_response(
        http_client,
        "POST",
        CHAT_COMPLETIONS_PATH,
        payload,
        recorder=recorder,
    )
    if response.status_code == 200:
        completion = response.json()
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"]["role"] == "assistant"
        return ContextProbeResult(
            repetitions=repetitions,
            status_code=response.status_code,
            response_text=response.text,
            outcome="success",
        )

    if is_context_overflow_response(response):
        return ContextProbeResult(
            repetitions=repetitions,
            status_code=response.status_code,
            response_text=response.text,
            outcome="overflow",
        )

    assert False, (
        f"Unexpected response while probing context length for {model!r} "
        f"at repetitions={repetitions}: "
        f"HTTP {response.status_code}: {response.text}"
    )


def find_search_bounds(
    http_client: httpx.Client,
    model: str,
    recorder: FailureArtifactRecorder,
) -> tuple[ContextProbeResult, ContextProbeResult]:
    best_success = probe_context_length(http_client, model, 0, recorder)
    assert best_success.outcome == "success"

    repetitions = 1
    for _ in range(MAX_EXPONENTIAL_SEARCH_STEPS):
        probe = probe_context_length(http_client, model, repetitions, recorder)
        if probe.outcome == "overflow":
            return best_success, probe
        best_success = probe
        repetitions *= 2

    assert False, (
        f"Failed to find an overflowing prompt for {model!r} within "
        f"{MAX_EXPONENTIAL_SEARCH_STEPS} exponential search steps."
    )


def binary_search_context_boundary(
    http_client: httpx.Client,
    model: str,
    low_probe: ContextProbeResult,
    high_probe: ContextProbeResult,
    recorder: FailureArtifactRecorder,
) -> tuple[ContextProbeResult, ContextProbeResult]:
    best_success = low_probe
    first_overflow = high_probe
    low_repetitions = low_probe.repetitions
    high_repetitions = high_probe.repetitions

    while low_repetitions + 1 < high_repetitions:
        mid_repetitions = (low_repetitions + high_repetitions) // 2
        probe = probe_context_length(
            http_client,
            model,
            mid_repetitions,
            recorder,
        )
        if probe.outcome == "success":
            best_success = probe
            low_repetitions = mid_repetitions
        else:
            first_overflow = probe
            high_repetitions = mid_repetitions

    return best_success, first_overflow


def test_context_length_finds_a_finite_boundary(
    http_client: httpx.Client,
    model: str,
    failure_artifact_recorder: FailureArtifactRecorder,
) -> None:
    assert_model_is_listed(
        http_client,
        model,
        failure_artifact_recorder,
    )
    low_probe, high_probe = find_search_bounds(
        http_client,
        model,
        failure_artifact_recorder,
    )
    best_success, first_overflow = binary_search_context_boundary(
        http_client,
        model,
        low_probe,
        high_probe,
        failure_artifact_recorder,
    )

    assert best_success.outcome == "success"
    assert first_overflow.outcome == "overflow"
    assert best_success.repetitions < first_overflow.repetitions
