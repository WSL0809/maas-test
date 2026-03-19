from __future__ import annotations

from pathlib import Path

from main import filter_cases, parse_test_run_markdown


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_main_case_filter_supports_prefix_letter() -> None:
    test_run_text = (REPO_ROOT / "test_run.md").read_text(encoding="utf-8")
    cases, _ = parse_test_run_markdown(test_run_text)

    selected_cases, warnings = filter_cases(cases, ["A"])
    selected_ids = [case.case_id for case in selected_cases]

    assert not warnings
    assert "A1" in selected_ids
    assert "A12" in selected_ids
    assert all(case_id.startswith("A") for case_id in selected_ids)


def test_main_case_filter_supports_prefix_glob() -> None:
    test_run_text = (REPO_ROOT / "test_run.md").read_text(encoding="utf-8")
    cases, _ = parse_test_run_markdown(test_run_text)

    selected_cases, warnings = filter_cases(cases, ["A*"])
    selected_ids = [case.case_id for case in selected_cases]

    assert not warnings
    assert "A1" in selected_ids
    assert "A12" in selected_ids
    assert all(case_id.startswith("A") for case_id in selected_ids)

