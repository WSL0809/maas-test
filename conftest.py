from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from chat_test_support import (
    DEFAULT_MATRIX_CONFIG_FILE,
    DEFAULT_MODEL,
    FailureArtifactRecorder,
    MatrixConfig,
    build_http_client,
    build_sdk_client,
    load_dotenv,
    load_matrix_config,
    split_model_names,
    unique_model_names,
)


load_dotenv()


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
    parser.addoption(
        "--run-sdk-smoke",
        action="store_true",
        default=False,
        help="Run tests marked as sdk_smoke.",
    )
    parser.addoption(
        "--run-strict-compat",
        action="store_true",
        default=False,
        help="Run tests marked as strict_compat.",
    )


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
    if config.getoption("run_sdk_smoke"):
        skip_sdk_smoke = None
    else:
        skip_sdk_smoke = pytest.mark.skip(reason="Pass --run-sdk-smoke to run SDK smoke tests.")

    if config.getoption("run_strict_compat"):
        skip_strict_compat = None
    else:
        skip_strict_compat = pytest.mark.skip(reason="Pass --run-strict-compat to run strict compatibility tests.")

    for item in items:
        model_name = getattr(getattr(item, "cls", None), "MODEL_NAME", None)
        if model_name and model_name not in selected_model_names:
            deselected_items.append(item)
            continue

        if skip_sdk_smoke is not None and "sdk_smoke" in item.keywords:
            item.add_marker(skip_sdk_smoke)
        if skip_strict_compat is not None and "strict_compat" in item.keywords:
            item.add_marker(skip_strict_compat)
        remaining_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = remaining_items
    else:
        items[:] = remaining_items

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or report.passed:
        return

    recorder = getattr(item, "_failure_artifact_recorder", None)
    if recorder is None or not recorder.exchanges:
        return

    artifact_path = recorder.write_failure_artifact(report.longreprtext)
    report.sections.append(("failure artifact", str(artifact_path)))
