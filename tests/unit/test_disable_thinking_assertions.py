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

    assert result.reasoning is None
    assert result.text == "The user is asking for the single word quartz. quartz"


def test_disable_thinking_assertion_accepts_exact_visible_answer() -> None:
    StrictDisableThinkingAssertions().assert_disable_thinking_reasoning_suppressed(
        None,
        " Quartz. ",
        transport="stream",
    )


def test_disable_thinking_assertion_rejects_hidden_thinking_in_visible_text() -> None:
    with pytest.raises(AssertionError, match="leaked hidden thinking"):
        StrictDisableThinkingAssertions().assert_disable_thinking_reasoning_suppressed(
            None,
            "The user is asking for the single word quartz.",
            transport="stream",
        )
