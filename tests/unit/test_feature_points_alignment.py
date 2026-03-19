from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_RUN_PATH = REPO_ROOT / "test_run.md"
FEATURE_POINTS_PATH = REPO_ROOT / "测试功能点.md"

CASE_HEADING_RE = re.compile(r"^##\s+([A-Z]\d+)\b", re.MULTILINE)
CASE_TABLE_ROW_RE = re.compile(r"^\|\s*([A-Z]\d+)\s*\|", re.MULTILINE)


def _extract_case_ids_from_test_run(text: str) -> set[str]:
    return {match.group(1) for match in CASE_HEADING_RE.finditer(text)}


def _extract_case_ids_from_feature_points(text: str) -> set[str]:
    return {match.group(1) for match in CASE_TABLE_ROW_RE.finditer(text)}


def test_test_run_case_ids_exist_in_feature_points_doc() -> None:
    test_run_text = TEST_RUN_PATH.read_text(encoding="utf-8")
    feature_points_text = FEATURE_POINTS_PATH.read_text(encoding="utf-8")

    test_run_case_ids = _extract_case_ids_from_test_run(test_run_text)
    feature_case_ids = _extract_case_ids_from_feature_points(feature_points_text)

    missing = sorted(test_run_case_ids - feature_case_ids)
    assert not missing, f"Cases present in test_run.md but missing from 测试功能点.md: {missing}"
