from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from openai import APIError, OpenAI
from pydantic import BaseModel, Field


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_COMPLETION_TOKENS = 32000
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
CHAT_COMPLETIONS_PATH = "chat/completions"
ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
DEFAULT_MATRIX_CONFIG_FILE = ROOT_DIR / "chat_models.json"
FAILURE_ARTIFACTS_DIR = ROOT_DIR / "test_failure_artifacts"
MODEL_OVERRIDE_ENV_KEYS = {
    "OPENAI_CHAT_TEST_MODEL",
    "OPENAI_CHAT_TEST_MODELS",
}


class MatrixConfig(BaseModel):
    models: tuple[str, ...]


class ParsedAnswer(BaseModel):
    word: str = Field(description="The echoed word.")
    length: int = Field(description="Character count for the echoed word.")


class FailureArtifactRecorder:
    def __init__(self, test_name: str, nodeid: str, model: str) -> None:
        self.test_name = test_name
        self.nodeid = nodeid
        self.model = model
        self.exchanges: list[dict[str, Any]] = []

    def add_http_exchange(
        self,
        *,
        request: httpx.Request,
        response: httpx.Response | None,
        request_payload: dict[str, Any],
        sse_events: list[str] | None = None,
        response_text_override: str | None = None,
        exception: BaseException | None = None,
    ) -> None:
        request_body = request.content.decode("utf-8", errors="replace") if request.content else ""
        response_text = None
        response_json = None
        response_headers = None
        status_code = None

        if response is not None:
            response_text = response_text_override
            if response_text is None:
                try:
                    response_text = response.text
                except httpx.ResponseNotRead:
                    response_text = None
            response_headers = redact_headers(response.headers)
            status_code = response.status_code
            if response_text_override is None:
                try:
                    response_json = response.json()
                except (ValueError, httpx.ResponseNotRead):
                    response_json = None

        self.exchanges.append(
            {
                "transport": "httpx",
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": redact_headers(request.headers),
                    "payload": request_payload,
                    "body_text": request_body,
                },
                "response": {
                    "status_code": status_code,
                    "headers": response_headers,
                    "json": response_json,
                    "text": response_text,
                    "sse_events": sse_events,
                },
                "exception": serialize_exception(exception),
            }
        )

    def add_sdk_exchange(
        self,
        *,
        api: str,
        request_payload: dict[str, Any],
        response: Any = None,
        stream_chunks: list[dict[str, Any]] | None = None,
        exception: BaseException | None = None,
    ) -> None:
        response_dump = None
        if response is not None:
            model_dump = getattr(response, "model_dump", None)
            response_dump = model_dump() if callable(model_dump) else response

        self.exchanges.append(
            {
                "transport": "openai_sdk",
                "request": {
                    "api": api,
                    "payload": request_payload,
                },
                "response": {
                    "model_dump": response_dump,
                    "stream_chunks": stream_chunks,
                },
                "exception": serialize_exception(exception),
            }
        )

    def write_failure_artifact(self, failure_text: str) -> Path:
        FAILURE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = f"{timestamp}-{slugify(self.model)}-{slugify(self.test_name)}.json"
        path = FAILURE_ARTIFACTS_DIR / filename
        payload = {
            "test": {
                "name": self.test_name,
                "nodeid": self.nodeid,
                "model": self.model,
                "created_at": timestamp,
            },
            "failure": {
                "message": failure_text,
            },
            "exchanges": self.exchanges,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def parse_env_file(path: Path = ENV_FILE) -> dict[str, str]:
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


def load_dotenv(path: Path = ENV_FILE) -> None:
    for key, value in parse_env_file(path).items():
        if key in MODEL_OVERRIDE_ENV_KEYS and key in os.environ:
            continue
        os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_base_url() -> str:
    base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_API_BASE_URL
    return base_url.rstrip("/") + "/"


def build_http_client() -> httpx.Client:
    return httpx.Client(
        base_url=resolve_base_url(),
        headers={
            "Authorization": f"Bearer {require_env('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(60.0),
    )


def build_sdk_client() -> OpenAI:
    client_kwargs = {"api_key": require_env("OPENAI_API_KEY")}
    base_url = resolve_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def build_client() -> OpenAI:
    return build_sdk_client()


def split_model_names(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [model.strip() for model in raw_value.split(",") if model.strip()]


def unique_model_names(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


@lru_cache(maxsize=None)
def load_matrix_config(path: str | Path = DEFAULT_MATRIX_CONFIG_FILE) -> MatrixConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    models = tuple(unique_model_names([str(model).strip() for model in data.get("models", []) if str(model).strip()]))
    if not models:
        raise ValueError(f"Matrix config {config_path} does not define any models.")

    return MatrixConfig(models=models)


def slugify(value: str) -> str:
    normalized = "".join(char if char.isalnum() else "-" for char in value)
    trimmed = normalized.strip("-").lower()
    return trimmed or "artifact"


def redact_headers(headers: httpx.Headers) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def serialize_exception(exception: BaseException | None) -> dict[str, Any] | None:
    if exception is None:
        return None

    data: dict[str, Any] = {
        "type": type(exception).__name__,
        "message": str(exception),
    }
    if isinstance(exception, APIError):
        data["status_code"] = getattr(exception, "status_code", None)
        data["body"] = getattr(exception, "body", None)
    return data


def iter_sse_data(response: httpx.Response) -> list[str]:
    events: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            events.append(line[5:].strip())
    return events


def extract_sse_data(lines: list[str]) -> list[str]:
    events: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            events.append(line[5:].strip())
    return events


def request_json(
    http_client: httpx.Client,
    path: str,
    payload: dict[str, Any],
    recorder: FailureArtifactRecorder | None = None,
) -> dict[str, Any]:
    request = http_client.build_request("POST", path, json=payload)
    try:
        response = http_client.send(request)
    except Exception as exc:
        if recorder is not None:
            recorder.add_http_exchange(request=request, response=None, request_payload=payload, exception=exc)
        raise
    if recorder is not None:
        recorder.add_http_exchange(request=request, response=response, request_payload=payload)
    assert response.status_code == 200, response.text
    return response.json()


def request_sse(
    http_client: httpx.Client,
    path: str,
    payload: dict[str, Any],
    recorder: FailureArtifactRecorder | None = None,
) -> tuple[httpx.Response, list[str], str]:
    request = http_client.build_request("POST", path, json=payload)
    response: httpx.Response | None = None
    try:
        response = http_client.send(request, stream=True)
        lines = list(response.iter_lines())
        raw_text = "\n".join(lines)
        events = extract_sse_data(lines)
        if recorder is not None:
            recorder.add_http_exchange(
                request=request,
                response=response,
                request_payload=payload,
                sse_events=events,
                response_text_override=raw_text,
            )
        return response, events, raw_text
    except Exception as exc:
        if recorder is not None:
            recorder.add_http_exchange(request=request, response=response, request_payload=payload, exception=exc)
        raise
    finally:
        if response is not None:
            response.close()
