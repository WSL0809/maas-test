from __future__ import annotations

from collections.abc import Iterable


CANONICAL_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "minimax-m2.5": (
        "minimax-m25",
        "minimax-2.5",
        "minimax_m25",
        "minimax_m2.5",
        "minimax2.5",
        "MiniMax2.5",
        "MiniMax-2.5",
    ),
}


def _alias_key(name: str) -> str:
    return name.strip().lower()


MODEL_NAME_CANONICALIZATION: dict[str, str] = {
    _alias_key(alias): canonical
    for canonical, aliases in CANONICAL_MODEL_ALIASES.items()
    for alias in (canonical, *aliases)
}


def canonicalize_model_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        return ""
    return MODEL_NAME_CANONICALIZATION.get(_alias_key(normalized), normalized)


def canonicalize_model_names(names: Iterable[str]) -> list[str]:
    normalized = [canonicalize_model_name(name) for name in names]
    return [name for name in normalized if name]


def unique_requested_model_names(names: Iterable[str]) -> list[str]:
    unique_names: list[str] = []
    seen_canonical_names: set[str] = set()
    for name in names:
        requested_name = name.strip()
        if not requested_name:
            continue
        canonical_name = canonicalize_model_name(requested_name)
        if canonical_name in seen_canonical_names:
            continue
        seen_canonical_names.add(canonical_name)
        unique_names.append(requested_name)
    return unique_names
