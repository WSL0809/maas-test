from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from tests.chat_test_support import (
    DEFAULT_MATRIX_CONFIG_FILE,
    DEFAULT_MODEL,
    FailureArtifactRecorder,
    MatrixConfig,
    build_http_client,
    build_sdk_client,
    load_dotenv,
    load_matrix_config,
    resolve_base_url,
    split_model_names,
    unique_model_names,
)
from tests.csv_report import CsvReportCollector, build_run_metadata, emit_terminal_summary


load_dotenv()


MODEL_CAPABILITY_GATES: dict[str, str] = {
    "test_stream_returns_tool_call": "SUPPORTS_STREAM_TOOL_CALL",
    "test_stream_tool_call_round_trip_returns_final_assistant_message": "SUPPORTS_STREAM_TOOL_ROUND_TRIP",
    "test_create_returns_repeated_same_tool_calls": "SUPPORTS_REPEATED_SAME_TOOL_CALL",
    "test_edit_tool_returns_valid_arguments": "SUPPORTS_EDIT_TOOL",
    "test_task_tool_returns_valid_arguments": "SUPPORTS_TASK_TOOL",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("chat-models")
    group.addoption(
        "--chat-model",
        action="append",
        default=[],
        dest="chat_models",
        help="Repeatable model name. Pass multiple times to run the suite against multiple models.",
    )
    group.addoption(
        "--chat-model-config",
        action="store",
        default=None,
        dest="chat_model_config",
        help="Path to the JSON config file that defines the default model list.",
    )
    group.addoption(
        "--OPENAI_BASE_URL",
        action="store",
        default=None,
        dest="openai_base_url",
        help="Override OPENAI_BASE_URL for this pytest run.",
    )
    group.addoption(
        "--OPENAI_API_KEY",
        action="store",
        default=None,
        dest="openai_api_key",
        help="Override OPENAI_API_KEY for this pytest run. Defaults to an empty string when omitted.",
    )
    parser.addoption(
        "--run-sdk-smoke",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Tests marked as sdk_smoke run by default.",
    )
    parser.addoption(
        "--run-tool-calling-probe",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. Dataset-driven tool-calling tests run by default.",
    )
    parser.addoption(
        "--csv-report-dir",
        action="store",
        default=None,
        help="Write readable CSV test reports to the given directory.",
    )


def pytest_configure(config: pytest.Config) -> None:
    base_url = config.getoption("openai_base_url")
    api_key = config.getoption("openai_api_key")

    if base_url is not None:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key is None:
        os.environ.setdefault("OPENAI_API_KEY", "")
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    report_dir = config.getoption("csv_report_dir")
    if report_dir:
        selected_models, _ = _resolve_target_models(config)
        metadata = build_run_metadata(resolve_base_url(), selected_models)
        setattr(config, "_csv_report_collector", CsvReportCollector(Path(report_dir).expanduser(), metadata))
    else:
        setattr(config, "_csv_report_collector", None)


def _resolve_matrix_config_path(pytest_config: pytest.Config) -> Path:
    raw_path = pytest_config.getoption("chat_model_config")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_MATRIX_CONFIG_FILE


def _resolve_target_models(pytest_config: pytest.Config) -> tuple[list[str], MatrixConfig]:
    config_path = _resolve_matrix_config_path(pytest_config)
    matrix_config = load_matrix_config(config_path)
    cli_models = unique_model_names(pytest_config.getoption("chat_models") or [])
    env_models = unique_model_names(split_model_names(os.getenv("OPENAI_CHAT_TEST_MODELS")))
    single_model = os.getenv("OPENAI_CHAT_TEST_MODEL")
    if single_model:
        models = cli_models or env_models or [single_model]
    else:
        models = cli_models or env_models or list(matrix_config.models) or [DEFAULT_MODEL]
    return models, matrix_config


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model" in metafunc.fixturenames:
        models, _ = _resolve_target_models(metafunc.config)
        metafunc.parametrize("model", models, ids=models, scope="session")


@pytest.fixture
def failure_artifact_recorder(request: pytest.FixtureRequest) -> FailureArtifactRecorder:
    model_name = getattr(getattr(request.node, "cls", None), "MODEL_NAME", None)
    if not model_name and "model" in request.fixturenames:
        model_name = request.getfixturevalue("model")
    if not model_name:
        model_name = "unknown"

    recorder = FailureArtifactRecorder(
        test_name=request.node.name,
        nodeid=request.node.nodeid,
        model=model_name,
    )
    setattr(request.node, "_failure_artifact_recorder", recorder)
    return recorder


@pytest.fixture(scope="session")
def http_client() -> Iterator[httpx.Client]:
    client = build_http_client()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def sdk_client() -> Iterator[OpenAI]:
    client = build_sdk_client()
    try:
        yield client
    finally:
        client.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected_models, _ = _resolve_target_models(config)
    selected_model_names = set(selected_models)

    remaining_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []

    for item in items:
        model_name = getattr(getattr(item, "cls", None), "MODEL_NAME", None)
        if model_name and model_name not in selected_model_names:
            deselected_items.append(item)
            continue

        if _should_deselect_for_model_capability(item):
            deselected_items.append(item)
            continue

        remaining_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = remaining_items
    else:
        items[:] = remaining_items


def _should_deselect_for_model_capability(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        case = callspec.params.get("case")
        model = callspec.params.get("model")
        should_skip_model = getattr(case, "should_skip_model", None)
        if callable(should_skip_model) and isinstance(model, str):
            return bool(should_skip_model(model))

    original_name = getattr(item, "originalname", item.name)
    required_capability = MODEL_CAPABILITY_GATES.get(original_name)
    test_class = getattr(item, "cls", None)
    if required_capability is not None and test_class is not None:
        return not bool(getattr(test_class, required_capability, True))

    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        recorder = getattr(item, "_failure_artifact_recorder", None)
        if recorder is not None and recorder.exchanges:
            artifact_path = recorder.write_failure_artifact(report.longreprtext)
            setattr(item, "_failure_artifact_path", artifact_path)
            report.sections.append(("failure artifact", str(artifact_path)))

    collector = getattr(item.config, "_csv_report_collector", None)
    if collector is not None:
        collector.record_result(item, report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    collector = getattr(session.config, "_csv_report_collector", None)
    if collector is None:
        return
    collector.write()


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    collector = getattr(config, "_csv_report_collector", None)
    if collector is None:
        return
    emit_terminal_summary(terminalreporter, collector)
