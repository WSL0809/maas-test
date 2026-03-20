from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_RUN_FILE = REPO_ROOT / "test_run.md"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "auto"
logger = logging.getLogger("maas-test.main")
KNOWN_MODEL_IDS = ("qwen35", "kimi-k25", "glm-5", "minimax-m21", "minimax-m25")
MODEL_TO_COLUMN = {
    "qwen35": "Qwen 3.5",
    "kimi-k25": "Kimi K2.5",
    "glm-5": "GLM-5",
    "minimax-m21": "Minimax 2.1",
    "minimax-m25": "Minimax 2.5",
}
MODEL_ALIASES = {
    "minimax-m21": {"minimax-m21", "minimax-m2.1"},
    "minimax-m25": {"minimax-m25", "minimax-m2.5"},
}
STATUS_PASS = "✅"
STATUS_FAIL = "❌"
STATUS_PARTIAL = "⚠️"
STATUS_NOT_RUN = "⏳"
CASE_HEADING_RE = re.compile(r"^##\s+([A-Z]\d+)\b(.*)$")
FENCE_START_RE = re.compile(r"^\s*```(?:bash|sh)\s*$")
FENCE_END_RE = re.compile(r"^\s*```\s*$")
LABEL_RE = re.compile(r"^\s*-\s+(.+?)\s*[：:]\s*$")
DEFAULT_CONNECTIVITY_TIMEOUT_SECONDS = 5.0


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
    artifact_dir: str
    results_csv: str
    stdout_file: str
    stderr_file: str
    exit_code: int | None
    duration_seconds: float
    error_message: str = ""
    statuses: dict[str, str] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute pytest commands from test_run.md and write a machine-readable run_manifest.json plus per-case artifacts."
    )
    parser.add_argument(
        "--test-run-file",
        default=str(DEFAULT_TEST_RUN_FILE),
        help="Path to the machine-readable test_run markdown file.",
    )
    parser.add_argument(
        "--template-file",
        default=None,
        help="Deprecated compatibility option. Ignored.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for case artifacts and run_manifest.json. A file path is treated as its parent directory.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        dest="openai_base_url",
        help="Override OPENAI_BASE_URL passed through to each pytest command.",
    )
    parser.add_argument(
        "--OPENAI_BASE_URL",
        default=None,
        dest="openai_base_url",
        help="Alias for --base-url. Override OPENAI_BASE_URL passed through to each pytest command.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="cases",
        help=(
            "Repeatable case filter, e.g. --case H1 --case B8. "
            "Prefix filters are supported: --case A selects all A* cases; --case A* also works."
        ),
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
    parser.add_argument(
        "--skip-connectivity-check",
        action="store_true",
        help="Skip the preflight connectivity probe against OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Stream each pytest command's stdout/stderr to the terminal while also writing stdout.txt/stderr.txt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for main.py execution progress.",
    )
    return parser.parse_args(argv)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        values[key] = value

    return values


def _load_dotenv(path: Path) -> None:
    for key, value in _parse_env_file(path).items():
        os.environ.setdefault(key, value)


def _normalize_base_url(base_url: str) -> tuple[str, str]:
    """
    Return (normalized_url, warning).

    Normalization ensures:
    - the URL is a valid http(s) URL
    - a /v1 suffix exists (OpenAI-compatible endpoints)
    - no trailing slash
    """
    raw = base_url.strip()
    if not raw:
        return "", "OPENAI_BASE_URL is empty"

    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return "", f"Invalid OPENAI_BASE_URL (expected http(s) URL): {base_url!r}"

    path = (parts.path or "").rstrip("/")
    warning = ""
    if not path.endswith("/v1"):
        warning = f"OPENAI_BASE_URL did not end with /v1; normalized from {base_url!r}"
        path = f"{path}/v1" if path else "/v1"

    normalized = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return normalized, warning


def _build_probe_url(normalized_base_url: str) -> str:
    return f"{normalized_base_url.rstrip('/')}/models"


def check_connectivity(
    *,
    openai_base_url: str | None,
    timeout_seconds: float = DEFAULT_CONNECTIVITY_TIMEOUT_SECONDS,
) -> tuple[bool, str, str | None]:
    """
    Probe the OpenAI-compatible endpoint before running live tests.

    Returns (ok, message, normalized_base_url_if_any).
    """
    _load_dotenv(REPO_ROOT / ".env")

    raw_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL", "")
    if not raw_base_url:
        return (
            False,
            "Missing OPENAI_BASE_URL. Set it in `.env` or pass `--OPENAI_BASE_URL/--base-url`.",
            None,
        )

    normalized_base_url, warning = _normalize_base_url(raw_base_url)
    if not normalized_base_url:
        return False, warning or f"Invalid OPENAI_BASE_URL: {raw_base_url!r}", None

    api_key = os.getenv("OPENAI_API_KEY", "")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    probe_url = _build_probe_url(normalized_base_url)
    try:
        import httpx
    except Exception as exc:
        return (
            False,
            f"Connectivity check requires httpx. Install dependencies via `uv sync` (import error: {exc}).",
            normalized_base_url,
        )

    timeout = httpx.Timeout(timeout_seconds, connect=timeout_seconds)
    try:
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
            response = client.get(probe_url)
    except httpx.RequestError as exc:
        return False, f"Failed to reach {probe_url!r}: {exc}", normalized_base_url

    if 200 <= response.status_code < 300:
        ok_message = f"Connectivity OK: {probe_url} -> {response.status_code}"
        if warning:
            ok_message = f"{ok_message} ({warning})"
        return True, ok_message, normalized_base_url

    body_snippet = (response.text or "").strip().replace("\n", " ")
    if len(body_snippet) > 200:
        body_snippet = body_snippet[:200] + "…"
    if response.status_code in {401, 403}:
        return (
            False,
            (
                f"Endpoint reachable but unauthorized: {probe_url} -> {response.status_code}. "
                "Set OPENAI_API_KEY (or adjust server auth)."
                + (f" Response: {body_snippet}" if body_snippet else "")
            ),
            normalized_base_url,
        )
    return (
        False,
        (
            f"Connectivity probe failed: {probe_url} -> {response.status_code}."
            + (f" Response: {body_snippet}" if body_snippet else "")
        ),
        normalized_base_url,
    )


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

    requested = [case_id.strip() for case_id in requested_cases if case_id.strip()]
    requested_exact = {case_id for case_id in requested if not _is_case_prefix_filter(case_id)}
    requested_prefixes = [_normalize_case_prefix(case_id) for case_id in requested if _is_case_prefix_filter(case_id)]
    requested_prefixes = [prefix for prefix in requested_prefixes if prefix]

    selected_cases: list[CaseDefinition] = []
    selected_case_ids: set[str] = set()
    for case in cases:
        if case.case_id in requested_exact:
            selected_cases.append(case)
            selected_case_ids.add(case.case_id)
            continue
        if requested_prefixes and any(case.case_id.startswith(prefix) for prefix in requested_prefixes):
            selected_cases.append(case)
            selected_case_ids.add(case.case_id)
            continue
    known_case_ids = {case.case_id for case in cases}
    warnings = [
        f"Requested case {case_id} was not found in test_run.md"
        for case_id in sorted(requested_exact - known_case_ids)
    ]
    for prefix in sorted(set(requested_prefixes)):
        if not any(case_id.startswith(prefix) for case_id in known_case_ids):
            warnings.append(f"Requested case prefix {prefix!r} matched no cases in test_run.md")
    return selected_cases, warnings


def _is_case_prefix_filter(case_id: str) -> bool:
    if not case_id:
        return False
    if case_id.endswith("*"):
        return True
    return len(case_id) == 1 and case_id.isalpha()


def _normalize_case_prefix(case_id: str) -> str:
    if not case_id:
        return ""
    if case_id.endswith("*"):
        case_id = case_id[:-1]
    return case_id.strip().upper()


def build_output_root(output_arg: str | None) -> tuple[Path, str]:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if output_arg:
        output_path = Path(output_arg).expanduser().resolve()
        output_root = output_path.parent if output_path.suffix else output_path
    else:
        output_root = (DEFAULT_REPORTS_DIR / timestamp).resolve()
    return output_root, timestamp


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def rewrite_command_args(
    command: str,
    report_dir: Path,
    chat_models: list[str] | None = None,
    openai_base_url: str | None = None,
) -> list[str]:
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
        if arg == "--OPENAI_BASE_URL":
            skip_next = True
            continue
        if arg.startswith("--OPENAI_BASE_URL="):
            continue
        rewritten.append(arg)

    for model in chat_models or []:
        rewritten.extend(["--chat-model", model])
    if openai_base_url:
        rewritten.extend(["--OPENAI_BASE_URL", openai_base_url])
    rewritten.extend(["--csv-report-dir", str(report_dir)])
    return rewritten


def execute_case(
    case: CaseDefinition,
    output_root: Path,
    chat_models: list[str] | None = None,
    openai_base_url: str | None = None,
    live_output: bool = False,
    *,
    index: int,
    total: int,
) -> CaseExecutionSummary:
    case_dir = output_root / "cases" / case.case_id
    results_csv_path = case_dir / "results.csv"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"
    case_dir.mkdir(parents=True, exist_ok=True)
    command_args = rewrite_command_args(
        case.selected_command.command,
        case_dir,
        chat_models,
        openai_base_url=openai_base_url,
    )
    start = time.perf_counter()
    logger.info(
        "[%s/%s] START %s %s",
        index,
        total,
        case.case_id,
        case.title,
    )
    logger.info("  label: %s", case.selected_command.label or "<unlabeled>")
    logger.info("  artifacts: %s", case_dir)
    logger.info("  command: %s", shlex.join(command_args))

    try:
        if live_output:
            exit_code, stdout_text, stderr_text = run_command_live(
                command_args,
                cwd=REPO_ROOT,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            error_message = ""
        else:
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
    if not live_output:
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
    elif exit_code is None and error_message:
        # If we failed to start the subprocess, still materialize the expected files.
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"{error_message}\n", encoding="utf-8")

    statuses, aggregation_error = aggregate_case_statuses(results_csv_path)
    if aggregation_error:
        error_message = f"{error_message}; {aggregation_error}".strip("; ")

    logger.info(
        "[%s/%s] END   %s exit=%s duration=%.2fs results=%s",
        index,
        total,
        case.case_id,
        exit_code,
        duration_seconds,
        results_csv_path,
    )
    if error_message:
        logger.warning("[%s/%s] %s error: %s", index, total, case.case_id, error_message)

    return CaseExecutionSummary(
        case_id=case.case_id,
        title=case.title,
        command=shlex.join(command_args),
        artifact_dir=str(case_dir),
        results_csv=str(results_csv_path),
        stdout_file=str(stdout_path),
        stderr_file=str(stderr_path),
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        error_message=error_message,
        statuses=statuses,
    )


def run_command_live(
    command_args: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str, str]:
    # Tee subprocess output to both files and the current terminal. We intentionally
    # don't keep output in memory; the returned strings are empty placeholders.
    import threading

    def pump(stream: Any, sink: Any, mirror: Any) -> None:
        try:
            for line in stream:
                sink.write(line)
                sink.flush()
                mirror.write(line)
                mirror.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        proc = subprocess.Popen(
            command_args,
            cwd=cwd,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        t_out = threading.Thread(
            target=pump,
            args=(proc.stdout, stdout_handle, sys.stdout),
            daemon=True,
        )
        t_err = threading.Thread(
            target=pump,
            args=(proc.stderr, stderr_handle, sys.stderr),
            daemon=True,
        )
        t_out.start()
        t_err.start()
        returncode = proc.wait()
        t_out.join(timeout=10)
        t_err.join(timeout=10)

    return returncode, "", ""


def aggregate_case_statuses(results_csv_path: Path) -> tuple[dict[str, str], str]:
    row_statuses = {
        "Qwen 3.5": STATUS_NOT_RUN,
        "Kimi K2.5": STATUS_NOT_RUN,
        "GLM-5": STATUS_NOT_RUN,
        "Minimax 2.1": STATUS_NOT_RUN,
        "Minimax 2.5": STATUS_NOT_RUN,
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
    return row_statuses, ""


def collapse_outcomes_for_model(rows: list[dict[str, str]], model_id: str) -> str:
    aliases = MODEL_ALIASES.get(model_id, {model_id})
    outcomes = [row["outcome"] for row in rows if row.get("model") in aliases]
    if not outcomes:
        return STATUS_NOT_RUN
    unique_outcomes = set(outcomes)
    if unique_outcomes == {"passed"}:
        return STATUS_PASS
    if unique_outcomes == {"failed"} or unique_outcomes == {"skipped"}:
        return STATUS_FAIL
    return STATUS_PARTIAL


def write_manifest(
    manifest_path: Path,
    *,
    timestamp: str,
    test_run_file: Path,
    output_root: Path,
    summaries: list[CaseExecutionSummary],
    warnings: list[str],
) -> None:
    manifest = {
        "schema": "maas-test.main-run-manifest",
        "version": 3,
        "timestamp": timestamp,
        "test_run_file": str(test_run_file),
        "output_root": str(output_root),
        "warnings": warnings,
        "cases": [asdict(summary) for summary in summaries],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def dry_run_output(
    cases: list[CaseDefinition],
    warnings: list[str],
    chat_models: list[str] | None = None,
    openai_base_url: str | None = None,
) -> str:
    lines = []
    for case in cases:
        rewritten_command = shlex.join(
            rewrite_command_args(
                case.selected_command.command,
                Path(f"/tmp/{case.case_id.lower()}-reports"),
                chat_models,
                openai_base_url=openai_base_url,
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
    configure_logging(args.verbose)
    test_run_file = Path(args.test_run_file).expanduser().resolve()

    try:
        test_run_text = test_run_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Failed to read test_run file: {exc}", file=sys.stderr)
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
        print(dry_run_output(cases, all_warnings, args.chat_models, openai_base_url=args.openai_base_url))
        return 0

    effective_base_url = args.openai_base_url
    if not args.skip_connectivity_check:
        ok, message, normalized = check_connectivity(openai_base_url=args.openai_base_url)
        if not ok:
            print(message, file=sys.stderr)
            return 1
        logger.info("%s", message)
        if effective_base_url is None and normalized:
            effective_base_url = normalized

    output_root, timestamp = build_output_root(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Selected %d case(s). Output root: %s", len(cases), output_root)
    if effective_base_url:
        logger.info("OPENAI_BASE_URL: %s", effective_base_url)
    if args.chat_models:
        logger.info("Chat model override(s): %s", ", ".join(args.chat_models))
    if args.live_output:
        logger.info("Live output: enabled (pytest stdout/stderr will stream to terminal)")
    for warning in all_warnings:
        logger.warning("%s", warning)

    summaries = [
        execute_case(
            case,
            output_root,
            args.chat_models,
            openai_base_url=effective_base_url,
            live_output=args.live_output,
            index=index,
            total=len(cases),
        )
        for index, case in enumerate(cases, start=1)
    ]
    runtime_warnings = [summary.error_message for summary in summaries if summary.error_message]
    manifest_path = output_root / "run_manifest.json"
    write_manifest(
        manifest_path,
        timestamp=timestamp,
        test_run_file=test_run_file,
        output_root=output_root,
        summaries=summaries,
        warnings=[*all_warnings, *runtime_warnings],
    )

    print(f"Artifacts written under {output_root}")
    print(f"Manifest written to {manifest_path}")

    if any(summary.exit_code not in (0, None) for summary in summaries):
        return 1
    if any(summary.exit_code is None for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
