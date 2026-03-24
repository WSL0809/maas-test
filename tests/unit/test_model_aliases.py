from __future__ import annotations

from model_aliases import unique_requested_model_names
from tests.chat_test_support import split_model_names, split_requested_model_names, unique_model_names


def test_split_model_names_canonicalizes_minimax_aliases() -> None:
    assert split_model_names("glm-5, MiniMax2.5, minimax_m25") == [
        "glm-5",
        "minimax-m2.5",
        "minimax-m2.5",
    ]


def test_unique_model_names_deduplicates_canonicalized_aliases() -> None:
    assert unique_model_names(["minimax-m2.5", "MiniMax2.5", " minimax_m2.5 "]) == [
        "minimax-m2.5",
    ]


def test_split_requested_model_names_preserves_requested_alias() -> None:
    assert split_requested_model_names("glm-5, MiniMax2.5, minimax_m25") == [
        "glm-5",
        "MiniMax2.5",
    ]


def test_unique_requested_model_names_preserves_first_requested_alias() -> None:
    assert unique_requested_model_names(["MiniMax2.5", "minimax-m2.5", " minimax_m2.5 "]) == [
        "MiniMax2.5",
    ]
