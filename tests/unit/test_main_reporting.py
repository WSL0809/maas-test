from __future__ import annotations

import csv
import json
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
uv run pytest -q tests/test_api_compatibility.py -k h1 --chat-model glm-5 --chat-model qwen35
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
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object --chat-model glm-5
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
            "--chat-model glm-5 --chat-model qwen35"
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


def test_rewrite_command_args_overrides_existing_openai_base_url_flag() -> None:
    rewritten = main.rewrite_command_args(
        "uv run pytest -q tests/test_chat.py --OPENAI_BASE_URL http://old.test/v1 -k test_json",
        Path("/tmp/new-reports"),
        openai_base_url="http://new.test/v1",
    )

    assert "--OPENAI_BASE_URL" in rewritten
    assert "http://old.test/v1" not in rewritten
    assert "http://new.test/v1" in rewritten


def test_aggregate_case_statuses_maps_models_and_merges_minimax(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    write_results_csv(
        results_csv,
        [
            {"model": "qwen35", "outcome": "passed"},
            {"model": "kimi-k25", "outcome": "failed"},
            {"model": "glm-5", "outcome": "passed"},
            {"model": "glm-5", "outcome": "skipped"},
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


def test_build_output_root_accepts_legacy_file_path(tmp_path: Path) -> None:
    output_root, _ = main.build_output_root(str(tmp_path / "reports" / "matrix_report.md"))

    assert output_root == (tmp_path / "reports").resolve()


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
    output_root = tmp_path / "reports"

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
                    {"model": "glm-5", "outcome": "passed"},
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
            "--output",
            str(output_root),
        ]
    )

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert not (output_root / "matrix_report.md").exists()
    assert manifest["schema"] == "maas-test.main-run-manifest"
    assert manifest["version"] == 2
    assert isinstance(manifest.get("output_root"), str) and manifest["output_root"]

    cases = manifest["cases"]
    assert [case["case_id"] for case in cases] == ["H1", "B8"]
    for case in cases:
        assert isinstance(case.get("artifact_dir"), str) and case["artifact_dir"]
        assert str(case["results_csv"]).endswith("/results.csv")
        assert str(case["stdout_file"]).endswith("/stdout.txt")
        assert str(case["stderr_file"]).endswith("/stderr.txt")


def test_dry_run_output_uses_effective_chat_model_override() -> None:
    cases = [
        main.CaseDefinition(
            case_id="H1",
            title="Chat",
            command_blocks=(
                main.CommandBlock(
                    label="全模型显式复测命令",
                    command="uv run pytest -q tests/test_api_compatibility.py --chat-model glm-5 --chat-model qwen35",
                ),
            ),
            selected_command=main.CommandBlock(
                label="全模型显式复测命令",
                command="uv run pytest -q tests/test_api_compatibility.py --chat-model glm-5 --chat-model qwen35",
            ),
        )
    ]

    output = main.dry_run_output(cases, [], ["kimi-k25"])

    assert "--chat-model kimi-k25" in output
    assert "--chat-model glm-5" not in output
    assert "--chat-model qwen35" not in output
