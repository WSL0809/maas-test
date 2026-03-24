from __future__ import annotations

import json

import pytest

from tests.chat_test_support import collect_stream_text
from tests.test_chat import BaseHTTPXChatTests


class StrictDisableThinkingAssertions(BaseHTTPXChatTests):
    __test__ = False

    MODEL_NAME = "demo-model"
    EXPECTS_REASONING_NULL_WHEN_THINKING_DISABLED = True


def test_collect_stream_text_keeps_content_out_of_reasoning_channel() -> None:
    events = [
        json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": "The user is asking for the single word quartz."},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": " quartz"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        "[DONE]",
    ]

    result = collect_stream_text(events)

    assert result.reasoning_text is None
    assert result.reasoning_content_text is None
    assert result.has_content is True
    assert result.text == "The user is asking for the single word quartz. quartz"


def test_collect_stream_text_preserves_all_three_channels() -> None:
    events = [
        json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning": "answer",
                            "reasoning_content": "answer",
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "choices": [
                    {
                        "delta": {"content": "2"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        "[DONE]",
    ]

    result = collect_stream_text(events)

    assert result.reasoning_text == "answer"
    assert result.reasoning_content_text == "answer"
    assert result.text == "2"
    assert result.has_reasoning is True
    assert result.has_reasoning_content is True
    assert result.has_content is True


def test_disable_thinking_stream_assertion_accepts_content_only() -> None:
    result = collect_stream_text(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": "quartz", "reasoning_content": None},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            "[DONE]",
        ]
    )

    StrictDisableThinkingAssertions().assert_stream_disable_thinking_channels_suppressed(result)


def test_disable_thinking_stream_assertion_rejects_reasoning_channels() -> None:
    result = collect_stream_text(
        [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning": "think",
                                "reasoning_content": "think",
                                "content": "quartz",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            "[DONE]",
        ]
    )

    with pytest.raises(AssertionError, match="leaked hidden thinking"):
        StrictDisableThinkingAssertions().assert_stream_disable_thinking_channels_suppressed(result)
