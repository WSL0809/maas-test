from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter


TEST_DESCRIPTION_REGISTRY = {
    "test_create_returns_non_empty_assistant_message": "验证基础 chat completion 返回非空 assistant 消息。",
    "test_create_respects_system_prompt_priority": "验证 system prompt 优先于冲突的 user 指令。",
    "test_create_preserves_multi_turn_context": "验证多轮对话历史可被保留，并用于回答后续问题。",
    "test_stream_sse_emits_content_and_done": "验证 SSE stream 返回文本内容，并以 [DONE] 事件结束。",
    "test_create_respects_max_completion_tokens_limit": "验证 max_completion_tokens 限制生效。",
    "test_create_supports_multilingual_output": "验证模型可按要求返回多语言文本。",
    "test_create_preserves_special_tokens_in_text": "验证特殊字符、代码和 HTML 片段可被保留。",
    "test_create_accepts_chat_template_kwargs_enable_thinking_false": (
        "验证 create 请求接受 chat_template_kwargs.enable_thinking=false。"
    ),
    "test_create_suppresses_reasoning_when_thinking_disabled": (
        "严格验证 create 请求在 enable_thinking=false 时不会返回 reasoning。"
    ),
    "test_stream_accepts_chat_template_kwargs_enable_thinking_false": (
        "验证 stream 请求接受 chat_template_kwargs.enable_thinking=false。"
    ),
    "test_stream_suppresses_reasoning_when_thinking_disabled": (
        "严格验证 stream 请求在 enable_thinking=false 时不会返回 reasoning。"
    ),
    "test_create_switches_thinking_to_instant_within_same_conversation": (
        "验证同一会话内 enable_thinking 从 true 切到 false 后仍可正常返回，并按 suppress reasoning 口径校验。"
    ),
    "test_create_switches_instant_to_thinking_within_same_conversation": (
        "验证同一会话内 enable_thinking 从 false 切到 true 后返回非空 reasoning。"
    ),
    "test_structured_output_tool_returns_valid_arguments": "验证 StructuredOutput 工具调用返回合法参数。",
    "test_json_mode_returns_valid_json_object": "验证 response_format=json_object 返回合法 JSON 对象。",
    "test_context_length_finds_a_finite_boundary": "探测上下文边界，并确认存在成功点和溢出点。",
    "test_sdk_create_returns_non_empty_assistant_message": "验证 OpenAI Python SDK 的基础 create 接入。",
    "test_sdk_stream_true_yields_non_empty_text": "验证 OpenAI Python SDK 的基础流式接入。",
    "test_multi_step_tool_chain_round_trip": "验证多步链式 tool loop 在 3 个工具步骤后生成最终文本。",
    "test_create_returns_tool_call": "验证基础 weather tool call 返回合法工具调用。",
    "test_stream_returns_tool_call": "验证流式 weather tool call 聚合后仍返回合法工具调用。",
    "test_tool_call_round_trip_returns_final_assistant_message": (
        "验证非流式 tool loop 回填工具结果后能生成最终 assistant 文本。"
    ),
    "test_stream_tool_call_round_trip_returns_final_assistant_message": (
        "验证流式 tool loop 回填工具结果后能生成最终 assistant 文本。"
    ),
    "test_create_returns_repeated_same_tool_calls": "验证同一响应内可连续返回两个同名工具调用。",
    "test_list_tool_returns_valid_arguments": "验证 OpenCode 风格 list 工具调用参数合法。",
    "test_read_tool_round_trip_returns_final_assistant_message": "验证 OpenCode 风格 read 工具两轮回填后仍能生成最终文本。",
    "test_grep_tool_returns_valid_arguments": "验证 OpenCode 风格 grep 工具调用参数合法。",
    "test_bash_tool_returns_valid_arguments": "验证 OpenCode 风格 bash 工具调用参数合法。",
    "test_edit_tool_returns_valid_arguments": "验证 OpenCode 风格 edit 工具调用参数合法。",
    "test_write_tool_returns_valid_arguments": "验证 OpenCode 风格 write 工具调用参数合法。",
    "test_task_tool_returns_valid_arguments": "验证 OpenCode 风格 task 工具调用参数合法。",
    "test_todowrite_tool_returns_valid_arguments": "验证 OpenCode 风格 todowrite 工具调用参数合法。",
    "test_create_accepts_multi_turn_history_with_assistant_and_tool_messages": (
        "验证包含 assistant 和 tool 历史消息的多轮会话形状可被接受。"
    ),
    "test_sdk_single_tool_call_returns_valid_json_arguments": "验证 SDK 非流式单工具调用返回合法 JSON 参数。",
    "test_sdk_stream_tool_call_emits_valid_json_arguments": "验证 SDK 流式单工具调用聚合后仍返回合法 JSON 参数。",
    "test_sdk_multi_tool_request_returns_valid_tool_calls": "验证 SDK 多工具请求返回合法工具调用。",
    "test_sdk_large_tool_arguments_remain_valid_json": "验证 SDK 路径下大参数工具调用仍保持 JSON 可解析。",
}

RESULTS_COLUMNS = (
    "run_id",
    "started_at",
    "suite",
    "test_name",
    "description",
    "nodeid",
    "model",
    "outcome",
    "duration_seconds",
    "base_url",
    "selected_models",
    "failure_summary",
    "failure_artifact",
)

SUMMARY_COLUMNS = (
    "测试类型",
    "case 名",
    "测试内容",
)

STATS_COLUMNS = (
    "run_id",
    "started_at",
    "suite",
    "model",
    "passed",
    "failed",
    "skipped",
    "total",
    "pass_rate",
)


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    started_at: str
    base_url: str
    selected_models: tuple[str, ...]


@dataclass(frozen=True)
class TestResultRow:
    run_id: str
    started_at: str
    suite: str
    test_name: str
    description: str
    nodeid: str
    model: str
    outcome: str
    duration_seconds: str
    base_url: str
    selected_models: str
    failure_summary: str
    failure_artifact: str


def build_run_metadata(base_url: str, selected_models: list[str]) -> RunMetadata:
    now = datetime.now().astimezone()
    started_at = now.replace(microsecond=0).isoformat()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    return RunMetadata(
        run_id=run_id,
        started_at=started_at,
        base_url=base_url,
        selected_models=tuple(selected_models),
    )


def build_test_description(test_name: str) -> str:
    description = TEST_DESCRIPTION_REGISTRY.get(test_name)
    if description is not None:
        return description

    raw = test_name[5:] if test_name.startswith("test_") else test_name
    humanized = " ".join(part for part in raw.split("_") if part)
    return humanized or test_name


def short_report_message(report: pytest.TestReport) -> str:
    if report.passed:
        return ""

    if report.skipped and isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
        message = str(report.longrepr[2])
        return " ".join(message.split())

    reprcrash = getattr(report.longrepr, "reprcrash", None)
    if reprcrash is not None:
        message = getattr(reprcrash, "message", "")
        if message:
            single_line = " ".join(str(message).split())
            if len(single_line) <= 240:
                return single_line
            return single_line[:237] + "..."

    lines = [line.strip() for line in report.longreprtext.splitlines() if line.strip()]
    if not lines:
        return report.outcome

    message = lines[-1]
    single_line = " ".join(message.split())
    if len(single_line) <= 240:
        return single_line
    return single_line[:237] + "..."


def should_record_report(report: pytest.TestReport) -> bool:
    if report.when == "call":
        return True
    if report.when == "setup" and (report.skipped or report.failed):
        return True
    if report.when == "teardown" and report.failed:
        return True
    return False


def resolve_item_model(item: pytest.Item) -> str:
    model_name = getattr(getattr(item, "cls", None), "MODEL_NAME", None)
    if model_name:
        return str(model_name)

    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        param_model = callspec.params.get("model")
        if param_model:
            return str(param_model)

    return "unknown"


class CsvReportCollector:
    def __init__(self, report_dir: Path, metadata: RunMetadata) -> None:
        self.report_dir = report_dir
        self.metadata = metadata
        self.results: list[TestResultRow] = []
        self.results_path: Path | None = None
        self.summary_path: Path | None = None
        self.stats_path: Path | None = None
        self._result_indexes: dict[str, int] = {}

    def record_result(self, item: pytest.Item, report: pytest.TestReport) -> None:
        if not should_record_report(report):
            return

        test_name = getattr(item, "originalname", None) or item.name
        suite = Path(item.location[0]).name
        model = resolve_item_model(item)
        failure_artifact = str(getattr(item, "_failure_artifact_path", "") or "")
        row = TestResultRow(
            run_id=self.metadata.run_id,
            started_at=self.metadata.started_at,
            suite=suite,
            test_name=test_name,
            description=build_test_description(test_name),
            nodeid=item.nodeid,
            model=model,
            outcome=report.outcome,
            duration_seconds=f"{report.duration:.3f}",
            base_url=self.metadata.base_url,
            selected_models=",".join(self.metadata.selected_models),
            failure_summary=short_report_message(report),
            failure_artifact=failure_artifact,
        )

        existing_index = self._result_indexes.get(item.nodeid)
        if existing_index is not None:
            if report.when == "teardown" and report.failed:
                self.results[existing_index] = row
            return

        self._result_indexes[item.nodeid] = len(self.results)
        self.results.append(row)

    def write(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.report_dir / "results.csv"
        self.summary_path = self.report_dir / "summary.csv"
        self.stats_path = self.report_dir / "stats.csv"
        self._write_results()
        self._write_summary()
        self._write_stats()

    def totals(self) -> dict[str, int]:
        counts = {"passed": 0, "failed": 0, "skipped": 0, "total": len(self.results)}
        for row in self.results:
            if row.outcome in counts:
                counts[row.outcome] += 1
        return counts

    def terminal_summary_lines(self) -> list[str]:
        totals = self.totals()
        return [
            f"CSV results: {self.results_path}",
            f"CSV summary: {self.summary_path}",
            f"CSV stats: {self.stats_path}",
            (
                "CSV totals: "
                f"passed={totals['passed']} failed={totals['failed']} "
                f"skipped={totals['skipped']} total={totals['total']}"
            ),
        ]

    def _write_results(self) -> None:
        assert self.results_path is not None
        with self.results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row.__dict__)

    def _write_summary(self) -> None:
        assert self.summary_path is not None
        summary_rows = self._build_summary_rows()
        fieldnames = list(SUMMARY_COLUMNS)
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    def _write_stats(self) -> None:
        assert self.stats_path is not None
        stats_rows = self._build_stats_rows()
        with self.stats_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATS_COLUMNS)
            writer.writeheader()
            writer.writerows(stats_rows)

    def _build_summary_rows(self) -> list[dict[str, str]]:
        cases: dict[tuple[str, str, str], dict[str, str]] = {}
        selected_models = set(self.metadata.selected_models)
        model_names = sorted(
            {
                row.model
                for row in self.results
                if row.model != "unknown" and (not selected_models or row.model in selected_models)
            }
        )

        for row in self.results:
            key = (row.suite, row.test_name, row.description)
            case_row = cases.setdefault(
                key,
                {
                    "测试类型": row.suite,
                    "case 名": row.test_name,
                    "测试内容": row.description,
                },
            )
            if row.model in model_names:
                case_row[f"{row.model}测试结果"] = row.outcome

        summary_rows: list[dict[str, str]] = []
        for suite, test_name, description in sorted(cases):
            case_row = cases[(suite, test_name, description)]
            for model_name in model_names:
                case_row.setdefault(f"{model_name}测试结果", "not_run")
            summary_rows.append(case_row)

        return summary_rows

    def _build_stats_rows(self) -> list[dict[str, str | int]]:
        counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0, "skipped": 0, "total": 0}
        )

        for row in self.results:
            self._increment_counts(counts[(row.suite, row.model)], row.outcome)
            self._increment_counts(counts[("ALL", row.model)], row.outcome)
            self._increment_counts(counts[("ALL", "ALL")], row.outcome)

        stats_rows: list[dict[str, str | int]] = []
        for suite, model in sorted(counts):
            bucket = counts[(suite, model)]
            total = bucket["total"]
            pass_rate = "0.00%" if total == 0 else f"{(bucket['passed'] / total) * 100:.2f}%"
            stats_rows.append(
                {
                    "run_id": self.metadata.run_id,
                    "started_at": self.metadata.started_at,
                    "suite": suite,
                    "model": model,
                    "passed": bucket["passed"],
                    "failed": bucket["failed"],
                    "skipped": bucket["skipped"],
                    "total": total,
                    "pass_rate": pass_rate,
                }
            )

        return stats_rows

    @staticmethod
    def _increment_counts(bucket: dict[str, int], outcome: str) -> None:
        bucket["total"] += 1
        if outcome in bucket:
            bucket[outcome] += 1


def emit_terminal_summary(terminalreporter: TerminalReporter, collector: CsvReportCollector) -> None:
    if collector.results_path is None or collector.summary_path is None:
        return

    terminalreporter.write_sep("-", "csv report")
    for line in collector.terminal_summary_lines():
        terminalreporter.write_line(line)
