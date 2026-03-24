from __future__ import annotations

from tests.chat_test_support import split_model_names, unique_model_names


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
