from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_RUN_FILE = REPO_ROOT / "test_run.md"
DEFAULT_TEMPLATE_FILE = REPO_ROOT / "draft.md"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "auto"
SUMMARY_STATUS_COLUMNS = ("Qwen 3.5", "Kimi K2.5", "GLM-5", "Minimax 2.1/2.5")
KNOWN_MODEL_IDS = ("qwen35", "kimi-k25", "glm-5", "minimax-m21", "minimax-m25")
MODEL_TO_COLUMN = {
    "qwen35": "Qwen 3.5",
    "kimi-k25": "Kimi K2.5",
    "glm-5": "GLM-5",
}
STATUS_PASS = "✅"
STATUS_FAIL = "❌"
STATUS_PARTIAL = "⚠️"
STATUS_NOT_RUN = "⏳"
CASE_HEADING_RE = re.compile(r"^##\s+([A-Z]\d+)\b(.*)$")
CASE_ROW_RE = re.compile(r"^\|\s*([A-Z]\d+)\s*\|")
FENCE_START_RE = re.compile(r"^\s*```(?:bash|sh)\s*$")
FENCE_END_RE = re.compile(r"^\s*```\s*$")
LABEL_RE = re.compile(r"^\s*-\s+(.+?)\s*[：:]\s*$")


@dataclass(frozen=True)
class CommandBlock:
    label: str
    command: str


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    title: str
    command_blocks: tuple[CommandBlock, ...]
    selected_command: CommandBlock


@dataclass
class CaseExecutionSummary:
    case_id: str
    title: str
    command: str
    report_dir: str
    exit_code: int | None
    duration_seconds: float
    error_message: str = ""
    statuses: dict[str, str] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute default pytest commands from test_run.md and render a matrix report."
    )
    parser.add_argument(
        "--test-run-file",
        default=str(DEFAULT_TEST_RUN_FILE),
        help="Path to the machine-readable test_run markdown file.",
    )
    parser.add_argument(
        "--template-file",
        default=str(DEFAULT_TEMPLATE_FILE),
        help="Path to the draft markdown template.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path. Defaults to reports/auto/<timestamp>/matrix_report.md.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help="Repeatable case filter, e.g. --case H1 --case B8.",
    )
    parser.add_argument(
        "--chat-model",
        action="append",
        default=[],
        dest="chat_models",
        help="Repeatable model override passed through to each pytest command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected cases and commands without executing them.",
    )
    return parser.parse_args(argv)


def parse_test_run_markdown(markdown_text: str) -> tuple[list[CaseDefinition], list[str]]:
    lines = markdown_text.splitlines()
    sections: list[tuple[str, str, list[str]]] = []
    warnings: list[str] = []
    current_case_id: str | None = None
    current_title = ""
    current_lines: list[str] = []

    def flush_current_section() -> None:
        if current_case_id is None:
            return
        sections.append((current_case_id, current_title, current_lines.copy()))

    for line in lines:
        heading_match = CASE_HEADING_RE.match(line)
        if heading_match:
            flush_current_section()
            current_case_id = heading_match.group(1)
            current_title = heading_match.group(2).strip()
            current_lines = []
            continue
        if current_case_id is not None:
            current_lines.append(line)

    flush_current_section()

    cases: list[CaseDefinition] = []
    for case_id, title, section_lines in sections:
        command_blocks = extract_command_blocks(section_lines)
        if not command_blocks:
            warnings.append(f"{case_id}: no runnable bash command found")
            continue
        selected_command, case_warnings = select_preferred_command(case_id, command_blocks)
        warnings.extend(case_warnings)
        if selected_command is None:
            warnings.append(f"{case_id}: no runnable command selected")
            continue
        cases.append(
            CaseDefinition(
                case_id=case_id,
                title=title,
                command_blocks=tuple(command_blocks),
                selected_command=selected_command,
            )
        )

    return cases, warnings


def extract_command_blocks(section_lines: list[str]) -> list[CommandBlock]:
    blocks: list[CommandBlock] = []
    last_label = ""
    index = 0

    while index < len(section_lines):
        line = section_lines[index]
        label_match = LABEL_RE.match(line)
        if label_match:
            last_label = normalize_label(label_match.group(1))
            index += 1
            continue

        if FENCE_START_RE.match(line):
            index += 1
            command_lines: list[str] = []
            while index < len(section_lines) and not FENCE_END_RE.match(section_lines[index]):
                command_lines.append(section_lines[index])
                index += 1
            blocks.append(CommandBlock(label=last_label, command="\n".join(command_lines).strip()))
            while index < len(section_lines) and not FENCE_END_RE.match(section_lines[index]):
                index += 1
            if index < len(section_lines):
                index += 1
            continue

        index += 1

    return [block for block in blocks if block.command]


def normalize_label(raw_label: str) -> str:
    return raw_label.strip().strip("`").strip()


def select_preferred_command(
    case_id: str,
    command_blocks: list[CommandBlock],
) -> tuple[CommandBlock | None, list[str]]:
    warnings: list[str] = []
    non_single_blocks = [block for block in command_blocks if "单模型" not in block.label]
    default_blocks = [block for block in non_single_blocks if "默认" in block.label]
    if default_blocks:
        if len(default_blocks) > 1:
            warnings.append(
                f"{case_id}: multiple default commands found; selected first label {default_blocks[0].label!r}"
            )
        return default_blocks[0], warnings

    full_blocks = [
        block
        for block in non_single_blocks
        if "全模型" in block.label or "显式复测" in block.label
    ]
    if full_blocks:
        return full_blocks[0], warnings

    if non_single_blocks:
        return non_single_blocks[0], warnings

    return (command_blocks[0] if command_blocks else None), warnings


def filter_cases(cases: list[CaseDefinition], requested_cases: list[str]) -> tuple[list[CaseDefinition], list[str]]:
    if not requested_cases:
        return cases, []

    requested = {case_id.strip() for case_id in requested_cases if case_id.strip()}
    selected_cases = [case for case in cases if case.case_id in requested]
    known_case_ids = {case.case_id for case in cases}
    warnings = [f"Requested case {case_id} was not found in test_run.md" for case_id in sorted(requested - known_case_ids)]
    return selected_cases, warnings


def build_output_paths(output_arg: str | None) -> tuple[Path, Path, str]:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if output_arg:
        output_path = Path(output_arg).expanduser().resolve()
        output_root = output_path.parent
    else:
        output_root = (DEFAULT_REPORTS_DIR / timestamp).resolve()
        output_path = output_root / "matrix_report.md"
    return output_root, output_path, timestamp


def rewrite_command_args(command: str, report_dir: Path, chat_models: list[str] | None = None) -> list[str]:
    args = shlex.split(command)
    rewritten: list[str] = []
    skip_next = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--csv-report-dir":
            skip_next = True
            continue
        if arg.startswith("--csv-report-dir="):
            continue
        if arg == "--chat-model":
            skip_next = True
            continue
        if arg.startswith("--chat-model="):
            continue
        rewritten.append(arg)

    for model in chat_models or []:
        rewritten.extend(["--chat-model", model])
    rewritten.extend(["--csv-report-dir", str(report_dir)])
    return rewritten


def execute_case(case: CaseDefinition, output_root: Path, chat_models: list[str] | None = None) -> CaseExecutionSummary:
    case_dir = output_root / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    command_args = rewrite_command_args(case.selected_command.command, case_dir, chat_models)
    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command_args,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        exit_code = completed.returncode
        error_message = ""
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    except OSError as exc:
        exit_code = None
        error_message = str(exc)
        stdout_text = ""
        stderr_text = ""

    duration_seconds = time.perf_counter() - start
    (case_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
    (case_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

    statuses, aggregation_error = aggregate_case_statuses(case_dir / "results.csv")
    if aggregation_error:
        error_message = f"{error_message}; {aggregation_error}".strip("; ")

    return CaseExecutionSummary(
        case_id=case.case_id,
        title=case.title,
        command=shlex.join(command_args),
        report_dir=str(case_dir),
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        error_message=error_message,
        statuses=statuses,
    )


def aggregate_case_statuses(results_csv_path: Path) -> tuple[dict[str, str], str]:
    row_statuses = {
        "Qwen 3.5": STATUS_NOT_RUN,
        "Kimi K2.5": STATUS_NOT_RUN,
        "GLM-5": STATUS_NOT_RUN,
        "Minimax 2.1/2.5": STATUS_NOT_RUN,
    }

    if not results_csv_path.exists():
        return row_statuses, f"missing results.csv at {results_csv_path}"

    try:
        with results_csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return row_statuses, f"failed to read results.csv: {exc}"

    model_statuses = {
        model_id: collapse_outcomes_for_model(rows, model_id)
        for model_id in KNOWN_MODEL_IDS
    }
    for model_id, column_name in MODEL_TO_COLUMN.items():
        row_statuses[column_name] = model_statuses[model_id]
    row_statuses["Minimax 2.1/2.5"] = merge_minimax_statuses(
        model_statuses["minimax-m21"],
        model_statuses["minimax-m25"],
    )
    return row_statuses, ""


def collapse_outcomes_for_model(rows: list[dict[str, str]], model_id: str) -> str:
    outcomes = [row["outcome"] for row in rows if row.get("model") == model_id]
    if not outcomes:
        return STATUS_NOT_RUN
    unique_outcomes = set(outcomes)
    if unique_outcomes == {"passed"}:
        return STATUS_PASS
    if unique_outcomes == {"failed"} or unique_outcomes == {"skipped"}:
        return STATUS_FAIL
    return STATUS_PARTIAL


def merge_minimax_statuses(minimax_m21_status: str, minimax_m25_status: str) -> str:
    if minimax_m21_status == minimax_m25_status and minimax_m21_status in {
        STATUS_PASS,
        STATUS_FAIL,
        STATUS_NOT_RUN,
    }:
        return minimax_m21_status
    return STATUS_PARTIAL


def render_report(
    template_text: str,
    summaries: list[CaseExecutionSummary],
    *,
    timestamp: str,
    test_run_file: Path,
    template_file: Path,
    output_root: Path,
    manifest_path: Path,
    warnings: list[str],
) -> str:
    summary_by_case = {summary.case_id: summary for summary in summaries}
    rendered_lines: list[str] = []

    for line in template_text.splitlines():
        rendered_lines.append(rewrite_case_row(line, summary_by_case))

    header_lines = build_generated_header(
        timestamp=timestamp,
        test_run_file=test_run_file,
        template_file=template_file,
        output_root=output_root,
        manifest_path=manifest_path,
    )

    if rendered_lines:
        insertion_index = 1
        while insertion_index < len(rendered_lines) and rendered_lines[insertion_index].strip():
            insertion_index += 1
        rendered_lines[insertion_index:insertion_index] = ["", *header_lines, ""]
    else:
        rendered_lines.extend(header_lines)

    rendered_lines.extend(["", *build_execution_summary_lines(summaries, warnings)])
    return "\n".join(rendered_lines).rstrip() + "\n"


def rewrite_case_row(line: str, summary_by_case: dict[str, CaseExecutionSummary]) -> str:
    if not CASE_ROW_RE.match(line):
        return line

    cells = split_markdown_row(line)
    if not cells:
        return line

    case_id = cells[0].strip()
    summary = summary_by_case.get(case_id)
    if summary is None or summary.statuses is None or len(cells) < 7:
        return line

    updated_cells = cells.copy()
    updated_cells[3] = summary.statuses["Qwen 3.5"]
    updated_cells[4] = summary.statuses["Kimi K2.5"]
    updated_cells[5] = summary.statuses["GLM-5"]
    updated_cells[6] = summary.statuses["Minimax 2.1/2.5"]
    return format_markdown_row(updated_cells)


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def format_markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_generated_header(
    *,
    timestamp: str,
    test_run_file: Path,
    template_file: Path,
    output_root: Path,
    manifest_path: Path,
) -> list[str]:
    return [
        "> 自动生成：本报告由 `main.py` 基于当前运行结果回填。",
        f"> 生成时间：`{timestamp}`",
        f"> 命令来源：`{test_run_file}`",
        f"> 模板来源：`{template_file}`",
        f"> 输出目录：`{output_root}`",
        f"> 运行清单：`{manifest_path}`",
    ]


def build_execution_summary_lines(
    summaries: list[CaseExecutionSummary],
    warnings: list[str],
) -> list[str]:
    lines = [
        "## 本次执行摘要",
        "",
        "| Case | Exit Code | Duration(s) | Raw Report |",
        "| --- | --- | --- | --- |",
    ]
    for summary in summaries:
        exit_code = "" if summary.exit_code is None else str(summary.exit_code)
        lines.append(
            f"| {summary.case_id} | {exit_code} | {summary.duration_seconds:.2f} | `{summary.report_dir}` |"
        )

    if warnings:
        lines.extend(["", "## 解析与运行告警", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    return lines


def write_manifest(
    manifest_path: Path,
    *,
    timestamp: str,
    test_run_file: Path,
    template_file: Path,
    output_file: Path,
    summaries: list[CaseExecutionSummary],
    warnings: list[str],
) -> None:
    manifest = {
        "timestamp": timestamp,
        "test_run_file": str(test_run_file),
        "template_file": str(template_file),
        "output_file": str(output_file),
        "warnings": warnings,
        "cases": [asdict(summary) for summary in summaries],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def dry_run_output(cases: list[CaseDefinition], warnings: list[str], chat_models: list[str] | None = None) -> str:
    lines = []
    for case in cases:
        rewritten_command = shlex.join(
            rewrite_command_args(
                case.selected_command.command,
                Path(f"/tmp/{case.case_id.lower()}-reports"),
                chat_models,
            )
        )
        lines.append(f"{case.case_id} {case.title}".rstrip())
        lines.append(f"  label: {case.selected_command.label or '<unlabeled>'}")
        lines.append(f"  command: {rewritten_command}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_run_file = Path(args.test_run_file).expanduser().resolve()
    template_file = Path(args.template_file).expanduser().resolve()

    try:
        test_run_text = test_run_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read test_run file: {exc}", file=sys.stderr)
        return 1

    try:
        template_text = template_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read template file: {exc}", file=sys.stderr)
        return 1

    cases, parse_warnings = parse_test_run_markdown(test_run_text)
    cases, filter_warnings = filter_cases(cases, args.cases)
    all_warnings = [*parse_warnings, *filter_warnings]

    if not cases:
        print("No runnable cases were selected.", file=sys.stderr)
        for warning in all_warnings:
            print(f"- {warning}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(dry_run_output(cases, all_warnings, args.chat_models))
        return 0

    output_root, output_path, timestamp = build_output_paths(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [execute_case(case, output_root, args.chat_models) for case in cases]
    runtime_warnings = [summary.error_message for summary in summaries if summary.error_message]
    manifest_path = output_root / "run_manifest.json"
    write_manifest(
        manifest_path,
        timestamp=timestamp,
        test_run_file=test_run_file,
        template_file=template_file,
        output_file=output_path,
        summaries=summaries,
        warnings=[*all_warnings, *runtime_warnings],
    )

    rendered_report = render_report(
        template_text,
        summaries,
        timestamp=timestamp,
        test_run_file=test_run_file,
        template_file=template_file,
        output_root=output_root,
        manifest_path=manifest_path,
        warnings=[*all_warnings, *runtime_warnings],
    )
    output_path.write_text(rendered_report, encoding="utf-8")

    print(f"Report written to {output_path}")
    print(f"Manifest written to {manifest_path}")

    if any(summary.exit_code not in (0, None) for summary in summaries):
        return 1
    if any(summary.exit_code is None for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
