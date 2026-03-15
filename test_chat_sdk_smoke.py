from __future__ import annotations

import pytest
from openai import OpenAI

from chat_test_support import DEFAULT_MAX_COMPLETION_TOKENS, FailureArtifactRecorder


pytestmark = pytest.mark.sdk_smoke


def test_sdk_create_returns_non_empty_assistant_message(
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
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
    }
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

    message = completion.choices[0].message

    assert completion.object == "chat.completion"
    assert completion.model
    assert message.role == "assistant"
    assert isinstance(message.content, str)
    assert message.content.strip()
    assert completion.usage is not None
    assert completion.usage.total_tokens > 0


def test_sdk_stream_true_yields_non_empty_text(
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
            {"role": "user", "content": "Reply with the word quartz."},
        ],
    }
    try:
        stream = sdk_client.chat.completions.create(**request_payload)
    except Exception as exc:
        failure_artifact_recorder.add_sdk_exchange(
            api="chat.completions.create",
            request_payload=request_payload,
            exception=exc,
        )
        raise

    parts: list[str] = []
    chunk_count = 0
    stream_chunks: list[dict] = []
    try:
        for chunk in stream:
            chunk_count += 1
            stream_chunks.append(chunk.model_dump())
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
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

    text = "".join(parts).strip()

    assert chunk_count > 0
    assert text
    assert "quartz" in text.lower()
