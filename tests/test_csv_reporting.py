from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest


pytest_plugins = ("pytester",)

REPO_ROOT = Path(__file__).resolve().parent.parent


def install_repo_plugin(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        textwrap.dedent(
            f"""
            import importlib.util
            import sys

            sys.path.insert(0, {str(REPO_ROOT)!r})

            spec = importlib.util.spec_from_file_location("repo_root_conftest", {str(REPO_ROOT / "tests" / "conftest.py")!r})
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            for name in dir(module):
                if name.startswith("pytest_") or name in {{"failure_artifact_recorder", "http_client", "sdk_client"}}:
                    globals()[name] = getattr(module, name)
            """
        )
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_csv_reports_are_opt_in(pytester: pytest.Pytester) -> None:
    install_repo_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
        def test_create_returns_non_empty_assistant_message():
            assert True
        """
    )

    result = pytester.runpytest("-q")

    result.assert_outcomes(passed=1)
    assert not (pytester.path / "reports").exists()


def test_csv_results_include_pass_fail_skip_and_artifact(pytester: pytest.Pytester) -> None:
    install_repo_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
        import httpx
        import pytest

        def test_create_returns_non_empty_assistant_message():
            assert True

        @pytest.mark.skip(reason="intentional skip")
        def test_sdk_create_returns_non_empty_assistant_message():
            assert True

        def test_stream_sse_emits_content_and_done(failure_artifact_recorder):
            request = httpx.Request("POST", "http://example.test/v1/chat/completions")
            response = httpx.Response(500, request=request, text="boom")
            failure_artifact_recorder.add_http_exchange(
                request=request,
                response=response,
                request_payload={"model": "demo"},
            )
            raise AssertionError("intentional failure")
        """
    )

    result = pytester.runpytest("-q", "--csv-report-dir=reports")

    result.assert_outcomes(passed=1, failed=1, skipped=1)

    rows = read_csv_rows(pytester.path / "reports" / "results.csv")
    assert [row["test_name"] for row in rows] == [
        "test_create_returns_non_empty_assistant_message",
        "test_sdk_create_returns_non_empty_assistant_message",
        "test_stream_sse_emits_content_and_done",
    ]

    pass_row, skip_row, fail_row = rows
    assert pass_row["outcome"] == "passed"
    assert pass_row["description"] == "验证基础 chat completion 返回非空 assistant 消息。"
    assert pass_row["failure_summary"] == ""
    assert pass_row["failure_artifact"] == ""

    assert skip_row["outcome"] == "skipped"
    assert skip_row["failure_summary"] == "Skipped: intentional skip"

    assert fail_row["outcome"] == "failed"
    assert fail_row["description"] == "验证 SSE stream 返回文本内容，并以 [DONE] 事件结束。"
    assert fail_row["failure_summary"] == "AssertionError: intentional failure"
    assert fail_row["failure_artifact"].endswith(".json")
    assert Path(fail_row["failure_artifact"]).exists()

    summary_rows = read_csv_rows(pytester.path / "reports" / "summary.csv")
    assert [row["case 名"] for row in summary_rows] == [
        "test_create_returns_non_empty_assistant_message",
        "test_sdk_create_returns_non_empty_assistant_message",
        "test_stream_sse_emits_content_and_done",
    ]
    assert all(list(row) == ["测试类型", "case 名", "测试内容"] for row in summary_rows)

    stats_rows = read_csv_rows(pytester.path / "reports" / "stats.csv")
    overall_row = next(row for row in stats_rows if row["suite"] == "ALL" and row["model"] == "ALL")
    assert overall_row["passed"] == "1"
    assert overall_row["failed"] == "1"
    assert overall_row["skipped"] == "1"
    assert overall_row["total"] == "3"


def test_csv_summary_aggregates_by_suite_and_model(pytester: pytest.Pytester) -> None:
    install_repo_plugin(pytester)
    pytester.makefile(
        ".json",
        chat_models=textwrap.dedent(
            """
            {
              "models": ["alpha", "beta"]
            }
            """
        ),
    )
    pytester.makepyfile(
        test_matrix="""
        def test_create_returns_non_empty_assistant_message(model):
            assert model in {"alpha", "beta"}
        """
    )

    result = pytester.runpytest("-q", "--csv-report-dir=reports", "--chat-model-config=chat_models.json")

    result.assert_outcomes(passed=2)

    rows = read_csv_rows(pytester.path / "reports" / "results.csv")
    assert [row["model"] for row in rows] == ["alpha", "beta"]
    assert all(row["selected_models"] == "alpha,beta" for row in rows)

    summary_rows = read_csv_rows(pytester.path / "reports" / "summary.csv")
    assert len(summary_rows) == 1
    assert list(summary_rows[0]) == ["测试类型", "case 名", "测试内容", "alpha测试结果", "beta测试结果"]
    assert summary_rows[0]["测试类型"] == "test_matrix.py"
    assert summary_rows[0]["case 名"] == "test_create_returns_non_empty_assistant_message"
    assert summary_rows[0]["测试内容"] == "验证基础 chat completion 返回非空 assistant 消息。"
    assert summary_rows[0]["alpha测试结果"] == "passed"
    assert summary_rows[0]["beta测试结果"] == "passed"

    stats_rows = read_csv_rows(pytester.path / "reports" / "stats.csv")
    lookup = {(row["suite"], row["model"]): row for row in stats_rows}
    assert lookup[("test_matrix.py", "alpha")]["passed"] == "1"
    assert lookup[("test_matrix.py", "beta")]["passed"] == "1"
    assert lookup[("ALL", "alpha")]["total"] == "1"
    assert lookup[("ALL", "beta")]["total"] == "1"
    assert lookup[("ALL", "ALL")]["total"] == "2"
    assert lookup[("ALL", "ALL")]["pass_rate"] == "100.00%"


def test_csv_summary_filters_unknown_model_columns(pytester: pytest.Pytester) -> None:
    install_repo_plugin(pytester)
    pytester.makefile(
        ".json",
        chat_models=textwrap.dedent(
            """
            {
              "models": ["alpha", "beta"]
            }
            """
        ),
    )
    pytester.makepyfile(
        test_mixed="""
        def test_create_returns_non_empty_assistant_message(model):
            assert model in {"alpha", "beta"}

        def test_sdk_create_returns_non_empty_assistant_message():
            assert True
        """
    )

    result = pytester.runpytest("-q", "--csv-report-dir=reports", "--chat-model-config=chat_models.json")

    result.assert_outcomes(passed=3)

    summary_rows = read_csv_rows(pytester.path / "reports" / "summary.csv")
    assert len(summary_rows) == 2
    assert "unknown测试结果" not in summary_rows[0]
    assert "unknown测试结果" not in summary_rows[1]
    assert list(summary_rows[0]) == ["测试类型", "case 名", "测试内容", "alpha测试结果", "beta测试结果"]
    assert list(summary_rows[1]) == ["测试类型", "case 名", "测试内容", "alpha测试结果", "beta测试结果"]
    row_by_case = {row["case 名"]: row for row in summary_rows}
    assert row_by_case["test_create_returns_non_empty_assistant_message"]["alpha测试结果"] == "passed"
    assert row_by_case["test_create_returns_non_empty_assistant_message"]["beta测试结果"] == "passed"
    assert row_by_case["test_sdk_create_returns_non_empty_assistant_message"]["alpha测试结果"] == "not_run"
    assert row_by_case["test_sdk_create_returns_non_empty_assistant_message"]["beta测试结果"] == "not_run"


def test_csv_excludes_deselected_items(pytester: pytest.Pytester) -> None:
    install_repo_plugin(pytester)
    pytester.makepyfile(
        test_models="""
        class TestAlpha:
            MODEL_NAME = "alpha"

            def test_create_returns_non_empty_assistant_message(self):
                assert True

        class TestBeta:
            MODEL_NAME = "beta"

            def test_create_returns_non_empty_assistant_message(self):
                assert True
        """
    )

    result = pytester.runpytest("-q", "--csv-report-dir=reports", "--chat-model=alpha")

    result.assert_outcomes(passed=1, deselected=1)

    rows = read_csv_rows(pytester.path / "reports" / "results.csv")
    assert len(rows) == 1
    assert rows[0]["model"] == "alpha"
    assert "beta" not in {row["model"] for row in rows}

    summary_rows = read_csv_rows(pytester.path / "reports" / "summary.csv")
    assert len(summary_rows) == 1
    assert summary_rows[0]["alpha测试结果"] == "passed"
    assert "beta测试结果" not in summary_rows[0]
