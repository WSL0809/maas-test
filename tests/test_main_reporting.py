from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import main


def write_results_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "outcome"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parser_selects_default_command_over_full_and_single_model() -> None:
    markdown = """
## H1 Chat Completions

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k h1 --chat-model glm5 --chat-model qwen35
```

- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k h1
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_api_compatibility.py -k h1 --chat-model kimi-k25
```
"""

    cases, warnings = main.parse_test_run_markdown(markdown)

    assert warnings == []
    assert len(cases) == 1
    assert cases[0].case_id == "H1"
    assert cases[0].selected_command.label == "默认模型矩阵复测命令"
    assert cases[0].selected_command.command == "uv run pytest -q tests/test_api_compatibility.py -k h1"


def test_parser_falls_back_to_explicit_command_when_default_missing() -> None:
    markdown = """
## B8 JSON Mode

- 显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object --chat-model glm5
```
"""

    cases, warnings = main.parse_test_run_markdown(markdown)

    assert warnings == []
    assert len(cases) == 1
    assert cases[0].selected_command.label == "显式复测命令"


def test_parser_warns_when_multiple_default_commands_exist() -> None:
    markdown = """
## B2 Thinking Off

- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k accepts
```

- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_chat.py -k suppresses
```
"""

    cases, warnings = main.parse_test_run_markdown(markdown)

    assert len(cases) == 1
    assert cases[0].selected_command.command == "uv run pytest -q tests/test_chat.py -k accepts"
    assert warnings == ["B2: multiple default commands found; selected first label '默认模型矩阵复测命令'"]


def test_rewrite_command_args_replaces_existing_csv_flag() -> None:
    rewritten = main.rewrite_command_args(
        "uv run pytest -q tests/test_chat.py --csv-report-dir old-reports -k test_json",
        Path("/tmp/new-reports"),
    )

    assert rewritten == [
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/test_chat.py",
        "-k",
        "test_json",
        "--csv-report-dir",
        "/tmp/new-reports",
    ]


def test_rewrite_command_args_overrides_existing_chat_model_flags() -> None:
    rewritten = main.rewrite_command_args(
        (
            "uv run pytest -q tests/test_api_compatibility.py -k h1 "
            "--chat-model glm5 --chat-model qwen35"
        ),
        Path("/tmp/new-reports"),
        ["kimi-k25"],
    )

    assert rewritten == [
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/test_api_compatibility.py",
        "-k",
        "h1",
        "--chat-model",
        "kimi-k25",
        "--csv-report-dir",
        "/tmp/new-reports",
    ]


def test_aggregate_case_statuses_maps_models_and_merges_minimax(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    write_results_csv(
        results_csv,
        [
            {"model": "qwen35", "outcome": "passed"},
            {"model": "kimi-k25", "outcome": "failed"},
            {"model": "glm5", "outcome": "passed"},
            {"model": "glm5", "outcome": "skipped"},
            {"model": "minimax-m21", "outcome": "passed"},
            {"model": "minimax-m25", "outcome": "passed"},
        ],
    )

    statuses, error = main.aggregate_case_statuses(results_csv)

    assert error == ""
    assert statuses == {
        "Qwen 3.5": "✅",
        "Kimi K2.5": "❌",
        "GLM-5": "⚠️",
        "Minimax 2.1/2.5": "✅",
    }


def test_render_report_updates_only_selected_case_rows() -> None:
    template = """# Matrix

## Section

| # | 测试点 | 测试内容 | Qwen 3.5 | Kimi K2.5 | GLM-5 | Minimax 2.1/2.5 | 优先级 |
|---|---|---|---|---|---|---|---|
| H1 | Chat | desc | ⏳ | ⏳ | ⏳ | ⏳ | P0 |
| H2 | Raw | desc | ✅ | ✅ | ✅ | ✅ | P1 |
"""
    summaries = [
        main.CaseExecutionSummary(
            case_id="H1",
            title="Chat",
            command="uv run pytest -q",
            report_dir="/tmp/reports/H1",
            exit_code=0,
            duration_seconds=1.25,
            statuses={
                "Qwen 3.5": "✅",
                "Kimi K2.5": "❌",
                "GLM-5": "⚠️",
                "Minimax 2.1/2.5": "✅",
            },
        )
    ]

    rendered = main.render_report(
        template,
        summaries,
        timestamp="20260319-120000",
        test_run_file=Path("test_run.md"),
        template_file=Path("draft.md"),
        output_root=Path("reports/auto/20260319-120000"),
        manifest_path=Path("reports/auto/20260319-120000/run_manifest.json"),
        warnings=[],
    )

    assert "> 自动生成：本报告由 `main.py` 基于当前运行结果回填。" in rendered
    assert "| H1 | Chat | desc | ✅ | ❌ | ⚠️ | ✅ | P0 |" in rendered
    assert "| H2 | Raw | desc | ✅ | ✅ | ✅ | ✅ | P1 |" in rendered
    assert "## 本次执行摘要" in rendered


def test_run_cli_end_to_end_with_synthetic_csv(tmp_path: Path, monkeypatch) -> None:
    test_run_file = tmp_path / "test_run.md"
    test_run_file.write_text(
        """
## H1 Chat

- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k h1
```

## B8 JSON

- 显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k b8
```
""",
        encoding="utf-8",
    )
    template_file = tmp_path / "draft.md"
    template_file.write_text(
        """# Matrix

| # | 测试点 | 测试内容 | Qwen 3.5 | Kimi K2.5 | GLM-5 | Minimax 2.1/2.5 | 优先级 |
|---|---|---|---|---|---|---|---|
| H1 | Chat | desc | ⏳ | ⏳ | ⏳ | ⏳ | P0 |
| B8 | JSON | desc | ⏳ | ⏳ | ⏳ | ⏳ | P0 |
| A1 | Base | desc | ✅ | ✅ | ✅ | ✅ | P0 |
""",
        encoding="utf-8",
    )
    output_file = tmp_path / "reports" / "matrix_report.md"

    def fake_run(args, cwd, text, capture_output, check):
        assert cwd == main.REPO_ROOT
        assert text is True
        assert capture_output is True
        assert check is False

        report_dir = Path(args[args.index("--csv-report-dir") + 1])
        case_id = report_dir.name
        if case_id == "H1":
            write_results_csv(
                report_dir / "results.csv",
                [
                    {"model": "qwen35", "outcome": "passed"},
                    {"model": "kimi-k25", "outcome": "passed"},
                    {"model": "glm5", "outcome": "passed"},
                    {"model": "minimax-m21", "outcome": "passed"},
                    {"model": "minimax-m25", "outcome": "failed"},
                ],
            )
        elif case_id == "B8":
            write_results_csv(
                report_dir / "results.csv",
                [
                    {"model": "qwen35", "outcome": "passed"},
                    {"model": "kimi-k25", "outcome": "passed"},
                ],
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    exit_code = main.run_cli(
        [
            "--test-run-file",
            str(test_run_file),
            "--template-file",
            str(template_file),
            "--output",
            str(output_file),
        ]
    )

    rendered = output_file.read_text(encoding="utf-8")
    manifest = (output_file.parent / "run_manifest.json").read_text(encoding="utf-8")

    assert exit_code == 0
    assert "| H1 | Chat | desc | ✅ | ✅ | ✅ | ⚠️ | P0 |" in rendered
    assert "| B8 | JSON | desc | ✅ | ✅ | ⏳ | ⏳ | P0 |" in rendered
    assert "| A1 | Base | desc | ✅ | ✅ | ✅ | ✅ | P0 |" in rendered
    assert '"case_id": "H1"' in manifest
    assert '"case_id": "B8"' in manifest


def test_dry_run_output_uses_effective_chat_model_override() -> None:
    cases = [
        main.CaseDefinition(
            case_id="H1",
            title="Chat",
            command_blocks=(
                main.CommandBlock(
                    label="全模型显式复测命令",
                    command="uv run pytest -q tests/test_api_compatibility.py --chat-model glm5 --chat-model qwen35",
                ),
            ),
            selected_command=main.CommandBlock(
                label="全模型显式复测命令",
                command="uv run pytest -q tests/test_api_compatibility.py --chat-model glm5 --chat-model qwen35",
            ),
        )
    ]

    output = main.dry_run_output(cases, [], ["kimi-k25"])

    assert "--chat-model kimi-k25" in output
    assert "--chat-model glm5" not in output
    assert "--chat-model qwen35" not in output
