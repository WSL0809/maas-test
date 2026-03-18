from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from httpcore import ConnectError as HttpcoreConnectError
from httpcore import ConnectTimeout as HttpcoreConnectTimeout
from httpcore import ReadError as HttpcoreReadError
from httpcore import RemoteProtocolError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from tests.chat_test_support import ROOT_DIR, get_api_key, load_dotenv, resolve_base_url

try:
    from jsonschema import ValidationError, validate
except ImportError:  # pragma: no cover - exercised via fallback validator tests
    ValidationError = ValueError
    validate = None

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm_asyncio = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional runtime dependency
    AutoTokenizer = None


logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_BASE_DELAY = 2.0
DEFAULT_RATE_LIMIT_MAX_DELAY = 60.0
DEFAULT_OUTPUT_FILE = "results.jsonl"
DEFAULT_SUMMARY_FILE = "summary.json"
DEFAULT_REPORT_ROOT = ROOT_DIR / "reports" / "k2vv"
HTTPX_STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=60.0)

ROLE_INPUT = "_input"
ROLE_SYSTEM = "system"

TOOL_CALLS_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_ARG_BEGIN = "<|tool_call_argument_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ID_PATTERN = re.compile(r"^functions\.(?P<name>[^:]+):(?P<index>\d+)$")
LOCAL_METADATA_KEY = "_meta"

RETRYABLE_READ_ERRORS = (
    HttpcoreConnectError,
    HttpcoreConnectTimeout,
    HttpcoreReadError,
    RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class SchemaValidationError(ValueError):
    pass


class _NullProgress:
    def __init__(self, total: int, desc: str = "", unit: str = "") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit

    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, count: int) -> None:
        return None


@dataclass
class StreamAccumulator:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    def add_chat_delta(
        self,
        *,
        content: str | None = None,
        reasoning: str | None = None,
        delta_tool_calls: Sequence[Any] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        if content:
            self.content_parts.append(content)
        if reasoning:
            self.reasoning_parts.append(reasoning)
        if delta_tool_calls:
            merge_delta_tool_calls(self.tool_calls, delta_tool_calls)
        if finish_reason:
            self.finish_reason = finish_reason
        if usage is not None:
            self.usage = usage

    def add_text(self, text: str | None) -> None:
        if text:
            self.content_parts.append(text)

    def build_response(
        self,
        *,
        request: Mapping[str, Any],
        request_id: str | None,
        created: int | None,
        use_raw_completions: bool,
    ) -> dict[str, Any]:
        content_text = "".join(self.content_parts)
        if use_raw_completions:
            extracted_tool_calls = extract_tool_call_info(content_text)
            if extracted_tool_calls:
                self.tool_calls = {i: tool_call for i, tool_call in enumerate(extracted_tool_calls)}
                self.finish_reason = "tool_calls"

        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "tool_calls": list(self.tool_calls.values()) if self.tool_calls else None,
        }
        reasoning_content_text = "".join(self.reasoning_parts) if self.reasoning_parts else None
        if reasoning_content_text:
            message_dict["reasoning_content"] = reasoning_content_text

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
            "usage": self.usage,
        }


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    description: str
    request: dict[str, Any]
    expected_finish_reason: str | None = None
    expected_tool_calls_valid: bool | None = None
    expected_tool_call_names: tuple[str, ...] = ()
    skip_models: tuple[str, ...] = ()
    data_index: int = 0

    @property
    def pytest_id(self) -> str:
        return self.case_id

    def should_skip_model(self, model: str) -> bool:
        return model in self.skip_models

    def to_dataset_entry(self) -> dict[str, Any]:
        entry = copy.deepcopy(self.request)
        metadata: dict[str, Any] = {
            "case_id": self.case_id,
            "description": self.description,
        }
        if self.expected_finish_reason is not None:
            metadata["expected_finish_reason"] = self.expected_finish_reason
        if self.expected_tool_calls_valid is not None:
            metadata["expected_tool_calls_valid"] = self.expected_tool_calls_valid
        if self.expected_tool_call_names:
            metadata["expected_tool_call_names"] = list(self.expected_tool_call_names)
        if self.skip_models:
            metadata["skip_models"] = list(self.skip_models)
        entry[LOCAL_METADATA_KEY] = metadata
        return entry


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def smart_exists(path: str | Path) -> bool:
    return Path(path).exists()


def smart_open(path: str | Path, mode: str, encoding: str = "utf-8"):
    file_path = Path(path)
    if any(flag in mode for flag in ("w", "a", "+", "x")):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path.open(mode, encoding=encoding)


def _progress(total: int, desc: str, unit: str):
    if tqdm_asyncio is None:
        return _NullProgress(total=total, desc=desc, unit=unit)
    return tqdm_asyncio(total=total, desc=desc, unit=unit)


def sanitize_tool_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return sanitized or "tool"


def is_normalized_tool_call_id(tool_call_id: str, tool_name: str) -> bool:
    match = TOOL_CALL_ID_PATTERN.fullmatch(tool_call_id)
    if match is None:
        return False
    return match.group("name") == sanitize_tool_name(tool_name)


def make_normalized_tool_call_id(tool_name: str, index: int) -> str:
    return f"functions.{sanitize_tool_name(tool_name)}:{index}"


def normalize_historical_tool_call_ids(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized

    tool_id_map: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue

        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                tool_name = str(function.get("name") or "tool")
                raw_id = str(tool_call.get("id") or "")
                normalized_id = raw_id
                if not raw_id or not is_normalized_tool_call_id(raw_id, tool_name):
                    normalized_id = make_normalized_tool_call_id(tool_name, index)
                if raw_id:
                    tool_id_map[raw_id] = normalized_id
                tool_call["id"] = normalized_id

        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id in tool_id_map:
                message["tool_call_id"] = tool_id_map[tool_call_id]

    return normalized


def split_dataset_request(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = copy.deepcopy(dict(payload))
    metadata = request.pop(LOCAL_METADATA_KEY, None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    return request, dict(metadata)


def _compute_backoff_delay(attempt: int) -> float:
    delay = min(DEFAULT_RATE_LIMIT_BASE_DELAY * (2**attempt), DEFAULT_RATE_LIMIT_MAX_DELAY)
    return delay + (delay * random.uniform(0, 0.25))


def _is_retryable_exception(e: BaseException) -> bool:
    if isinstance(e, RateLimitError):
        return True
    if isinstance(e, APIStatusError):
        return getattr(e, "status_code", None) == 429
    if isinstance(e, (APIConnectionError, APITimeoutError, *RETRYABLE_READ_ERRORS)):
        return True
    return False


def _serialize_error(e: BaseException) -> dict[str, str]:
    return {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "error": str(e),
    }


def extract_tool_call_info(tool_call_rsp: str) -> list[dict[str, Any]]:
    if TOOL_CALLS_BEGIN not in tool_call_rsp:
        return []

    section_pattern = rf"{re.escape(TOOL_CALLS_BEGIN)}(.*?){re.escape(TOOL_CALLS_END)}"
    tool_calls_sections = re.findall(section_pattern, tool_call_rsp, re.DOTALL)
    if not tool_calls_sections:
        return []

    func_call_pattern = (
        rf"{re.escape(TOOL_CALL_BEGIN)}\s*"
        r"(?P<tool_call_id>[\w\.]+:\d+)\s*"
        rf"{re.escape(TOOL_CALL_ARG_BEGIN)}\s*"
        r"(?P<function_arguments>.*?)\s*"
        rf"{re.escape(TOOL_CALL_END)}"
    )

    tool_calls: list[dict[str, Any]] = []
    for match in re.finditer(func_call_pattern, tool_calls_sections[0], re.DOTALL):
        function_id = match.group("tool_call_id")
        function_args = match.group("function_arguments")
        try:
            function_name = function_id.split(".")[1].split(":")[0]
        except IndexError:
            logger.warning("Unable to parse function_id: %s", function_id)
            continue

        tool_calls.append(
            {
                "id": function_id,
                "type": "function",
                "function": {"name": function_name, "arguments": function_args},
            }
        )

    return tool_calls


def compute_hash(obj: dict[str, Any]) -> str:
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def _read_attr_or_key(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def merge_delta_tool_calls(collected: dict[int, dict[str, Any]], delta_tool_calls: Sequence[Any]) -> None:
    for partial in delta_tool_calls:
        raw_index = _read_attr_or_key(partial, "index")
        index = raw_index if isinstance(raw_index, int) else 0
        current = collected.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )

        partial_id = _read_attr_or_key(partial, "id")
        if isinstance(partial_id, str) and partial_id:
            current["id"] = partial_id

        partial_type = _read_attr_or_key(partial, "type")
        if isinstance(partial_type, str) and partial_type:
            current["type"] = partial_type

        function = _read_attr_or_key(partial, "function")
        if function is None:
            continue

        name = _read_attr_or_key(function, "name")
        if isinstance(name, str) and name:
            current["function"]["name"] = name

        arguments = _read_attr_or_key(function, "arguments")
        if isinstance(arguments, str) and arguments:
            current["function"]["arguments"] += arguments


def _matches_type(instance: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(instance, Mapping)
    if expected_type == "array":
        return isinstance(instance, list)
    if expected_type == "string":
        return isinstance(instance, str)
    if expected_type == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected_type == "number":
        return (isinstance(instance, int) or isinstance(instance, float)) and not isinstance(instance, bool)
    if expected_type == "boolean":
        return isinstance(instance, bool)
    if expected_type == "null":
        return instance is None
    return True


def _type_name(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, Mapping):
        return "object"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    return type(instance).__name__


def _fallback_validate(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        raise SchemaValidationError(f"{path}: expected one of {enum}, got {instance!r}")

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: expected {schema['const']!r}, got {instance!r}")

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        errors: list[str] = []
        for candidate in any_of:
            try:
                _fallback_validate(instance, candidate, path)
                break
            except SchemaValidationError as exc:
                errors.append(str(exc))
        else:
            raise SchemaValidationError(f"{path}: none of anyOf matched ({'; '.join(errors)})")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = 0
        for candidate in one_of:
            try:
                _fallback_validate(instance, candidate, path)
                matches += 1
            except SchemaValidationError:
                continue
        if matches != 1:
            raise SchemaValidationError(f"{path}: expected exactly one oneOf match, got {matches}")

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for candidate in all_of:
            _fallback_validate(instance, candidate, path)

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _matches_type(instance, expected_type):
            raise SchemaValidationError(f"{path}: expected {expected_type}, got {_type_name(instance)}")
    elif isinstance(expected_type, list) and expected_type:
        if not any(_matches_type(instance, item) for item in expected_type if isinstance(item, str)):
            raise SchemaValidationError(f"{path}: expected one of {expected_type}, got {_type_name(instance)}")

    if isinstance(instance, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            raise SchemaValidationError(f"{path}: expected minLength {min_length}, got {len(instance)}")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(instance) > max_length:
            raise SchemaValidationError(f"{path}: expected maxLength {max_length}, got {len(instance)}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            raise SchemaValidationError(f"{path}: string did not match pattern {pattern!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            raise SchemaValidationError(f"{path}: expected minimum {minimum}, got {instance}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and instance > maximum:
            raise SchemaValidationError(f"{path}: expected maximum {maximum}, got {instance}")

    if isinstance(instance, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            raise SchemaValidationError(f"{path}: expected minItems {min_items}, got {len(instance)}")
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(instance) > max_items:
            raise SchemaValidationError(f"{path}: expected maxItems {max_items}, got {len(instance)}")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(instance):
                _fallback_validate(item, item_schema, f"{path}[{index}]")

    if isinstance(instance, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            required = schema.get("required")
            if isinstance(required, list):
                for key in required:
                    if key not in instance:
                        raise SchemaValidationError(f"{path}: missing required property {key!r}")

            for key, property_schema in properties.items():
                if key in instance and isinstance(property_schema, Mapping):
                    _fallback_validate(instance[key], property_schema, f"{path}.{key}")

            additional_properties = schema.get("additionalProperties", True)
            if additional_properties is False:
                extras = [key for key in instance if key not in properties]
                if extras:
                    raise SchemaValidationError(f"{path}: unexpected properties {extras}")
            elif isinstance(additional_properties, Mapping):
                for key, value in instance.items():
                    if key not in properties:
                        _fallback_validate(value, additional_properties, f"{path}.{key}")


def validate_tool_call_against_tools(tool_call: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]) -> bool:
    try:
        tool_name = tool_call["function"]["name"]
        schema = next(
            (
                t["function"]["parameters"]
                for t in tools
                if t["function"]["name"] == tool_name
            ),
            None,
        )
        if not schema:
            logger.warning("No schema found for tool '%s'", tool_name)
            return False

        arguments = tool_call["function"]["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed for tool '%s' arguments: %s", tool_name, exc)
                return False

        if validate is not None:
            validate(instance=arguments, schema=schema)
        else:
            _fallback_validate(arguments, schema)
        return True
    except ValidationError as exc:
        logger.warning("Schema validation failed for tool '%s': %s", tool_name, getattr(exc, "message", exc))
        return False
    except SchemaValidationError as exc:
        logger.warning("Schema validation failed for tool '%s': %s", tool_name, exc)
        return False
    except KeyError as exc:
        logger.warning("Tool call format error, missing field: %s", exc)
        return False
    except Exception as exc:
        logger.warning("Unexpected error during validation: %s", exc)
        return False


def build_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    model: str,
    eval_started_at: str | None,
    eval_finished_at: str | None,
    eval_duration_ms: int | None,
) -> dict[str, Any]:
    summary = {
        "model": model,
        "success_count": 0,
        "failure_count": 0,
        "finish_stop": 0,
        "finish_tool_calls": 0,
        "finish_others": 0,
        "finish_others_detail": {},
        "schema_validation_error_count": 0,
        "successful_tool_call_count": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "eval_started_at": eval_started_at,
        "eval_finished_at": eval_finished_at,
        "eval_duration_ms": eval_duration_ms,
    }

    for result in results:
        status = result.get("status")
        finish_reason = result.get("finish_reason")
        tool_calls_valid = result.get("tool_calls_valid")

        usage = (result.get("response") or {}).get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, int):
                    summary["usage"][key] += value

        if status == "success":
            summary["success_count"] += 1
        else:
            summary["failure_count"] += 1

        if finish_reason == "stop":
            summary["finish_stop"] += 1
        elif finish_reason == "tool_calls":
            summary["finish_tool_calls"] += 1
            if tool_calls_valid:
                summary["successful_tool_call_count"] += 1
            else:
                summary["schema_validation_error_count"] += 1
        elif finish_reason:
            summary["finish_others"] += 1
            summary["finish_others_detail"].setdefault(finish_reason, 0)
            summary["finish_others_detail"][finish_reason] += 1

    return summary


def load_dataset_cases(file_path: str | Path) -> list[DatasetCase]:
    if not smart_exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")

    cases: list[DatasetCase] = []
    with smart_open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            raw_entry = json.loads(line)
            request, metadata = split_dataset_request(raw_entry)
            skip_models = tuple(
                str(model).strip()
                for model in metadata.get("skip_models", [])
                if str(model).strip()
            )
            expected_finish_reason = metadata.get("expected_finish_reason")
            expected_tool_calls_valid = metadata.get("expected_tool_calls_valid")
            expected_tool_call_names = tuple(
                str(name).strip()
                for name in metadata.get("expected_tool_call_names", [])
                if str(name).strip()
            )
            description = str(metadata.get("description") or metadata.get("case_id") or f"case-{line_num}")
            case_id = str(metadata.get("case_id") or f"line-{line_num}")

            cases.append(
                DatasetCase(
                    case_id=case_id,
                    description=description,
                    request=request,
                    expected_finish_reason=(
                        str(expected_finish_reason) if expected_finish_reason is not None else None
                    ),
                    expected_tool_calls_valid=(
                        bool(expected_tool_calls_valid)
                        if expected_tool_calls_valid is not None
                        else None
                    ),
                    expected_tool_call_names=expected_tool_call_names,
                    skip_models=skip_models,
                    data_index=line_num,
                )
            )

    return cases


def build_default_report_paths(
    output_path: str | None = None,
    summary_path: str | None = None,
) -> tuple[Path, Path]:
    if output_path and summary_path:
        return Path(output_path), Path(summary_path)
    if output_path:
        output = Path(output_path)
        return output, output.with_name(DEFAULT_SUMMARY_FILE)
    if summary_path:
        summary = Path(summary_path)
        return summary.with_name(DEFAULT_OUTPUT_FILE), summary

    report_dir = DEFAULT_REPORT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    return report_dir / DEFAULT_OUTPUT_FILE, report_dir / DEFAULT_SUMMARY_FILE


def parse_extra_body(extra_body_text: str | None) -> dict[str, Any]:
    if not extra_body_text:
        return {}
    try:
        value = json.loads(extra_body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse --extra-body JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--extra-body must decode to a JSON object.")
    return value


def resolve_cli_base_url(raw_base_url: str | None) -> str:
    return (raw_base_url or resolve_base_url()).rstrip("/")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM Tool Calls Validator\n\n"
            "Validate LLM tool call functionality via HTTP API with concurrency support "
            "and optional incremental re-run.\n"
            "Each line in the test set file must be a complete LLM request body (JSON format)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("file_path", help="Test set file path (JSONL format)")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API endpoint URL, e.g., https://api.moonshot.cn/v1. Defaults to OPENAI_BASE_URL/.env.",
    )
    parser.add_argument("--api-key", help="API key (can also be set via OPENAI_API_KEY env var)")
    parser.add_argument("--model", required=True, help="Model name, e.g., kimi-k25")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides request temperature)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count (overrides request max_tokens)",
    )
    parser.add_argument("--extra-body", type=str, help="Extra request body parameters (JSON string)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Detailed results output file path (defaults to reports/k2vv/<timestamp>/results.jsonl)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Aggregated summary output file path (defaults to reports/k2vv/<timestamp>/summary.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Number of retries on failure (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only rerun failed or new requests, preserve successful results",
    )
    parser.add_argument(
        "--use_raw_completions",
        action="store_true",
        help="Use /v1/completions endpoint (requires tokenizer)",
    )
    parser.add_argument("--tokenizer-model", type=str, help="Tokenizer model name for raw completions")
    parser.add_argument(
        "--disable-tool-call-id-normalization",
        action="store_true",
        help="Do not rewrite historical tool call IDs to the functions.<name>:<idx> format",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


class ToolCallsValidator:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        output_file: str | Path = DEFAULT_OUTPUT_FILE,
        summary_file: str | Path = DEFAULT_SUMMARY_FILE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        extra_body: dict[str, Any] | None = None,
        incremental: bool = False,
        use_raw_completions: bool = False,
        tokenizer_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        normalize_tool_call_ids: bool = True,
    ):
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {max_retries}")
        if temperature is not None and (temperature < 0 or temperature > 1):
            raise ValueError(f"temperature must be between 0 and 1, got {temperature}")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key or get_api_key()
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.output_file = str(output_file)
        self.summary_file = str(summary_file)
        self.incremental = incremental
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_raw_completions = use_raw_completions
        self.tokenizer_model = tokenizer_model or model
        self.normalize_tool_call_ids = normalize_tool_call_ids

        self.results: list[dict[str, Any]] = []
        self.finish_reason_stat: dict[str, int] = {}
        self.eval_start_ts: float | None = None
        self.eval_end_ts: float | None = None
        self.eval_started_at: str | None = None
        self.eval_finished_at: str | None = None

        self.http_client = httpx.AsyncClient(
            timeout=HTTPX_STREAM_TIMEOUT,
            limits=httpx.Limits(
                max_connections=concurrency * 2,
                max_keepalive_connections=concurrency,
            ),
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
        )

        self.file_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()

        if use_raw_completions:
            if AutoTokenizer is None:
                raise RuntimeError(
                    "Raw completions mode requires the optional `transformers` dependency."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, trust_remote_code=True)
        else:
            self.tokenizer = None

        logger.info("Model: %s", self.model)
        logger.info("Results will be saved to: %s", self.output_file)
        logger.info("Summary will be saved to: %s", self.summary_file)
        logger.info("Concurrency: %s", self.concurrency)
        endpoint = "/v1/completions" if self.use_raw_completions else "/v1/chat/completions"
        logger.info("Request endpoint: %s", endpoint)
        if self.incremental:
            logger.info("Incremental mode: enabled")
        if self.normalize_tool_call_ids:
            logger.info("Historical tool call ID normalization: enabled")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.client.close()
            logger.debug("AsyncOpenAI client closed successfully")
        except Exception as e:
            logger.warning("Error closing AsyncOpenAI client: %s", e)
        try:
            await self.http_client.aclose()
        except Exception as e:
            logger.warning("Error closing httpx client: %s", e)
        return False

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_body, _ = split_dataset_request(request)
        req = (
            normalize_historical_tool_call_ids(request_body)
            if self.normalize_tool_call_ids
            else copy.deepcopy(request_body)
        )

        if "messages" in req:
            for message in req["messages"]:
                if message.get("role") == ROLE_INPUT:
                    message["role"] = ROLE_SYSTEM

        if self.model:
            req["model"] = self.model

        if self.temperature is not None:
            req["temperature"] = self.temperature
        if self.max_tokens is not None:
            if "max_completion_tokens" in req and "max_tokens" not in req:
                req["max_completion_tokens"] = self.max_tokens
            else:
                req["max_tokens"] = self.max_tokens

        if req.get("stream", False) and not self.use_raw_completions:
            so = req.get("stream_options")
            if not isinstance(so, dict):
                so = {}
            so.setdefault("include_usage", True)
            req["stream_options"] = so

        if self.use_raw_completions and self.tokenizer:
            if "max_completion_tokens" in req and "max_tokens" not in req:
                req["max_tokens"] = req.pop("max_completion_tokens")
            req["prompt"] = self.tokenizer.apply_chat_template(
                req["messages"],
                tokenize=False,
                tools=req.get("tools", None),
                add_generation_prompt=True,
            )
            req.pop("messages")
            if "tools" in req:
                req.pop("tools")

        return req

    def read_jsonl(self, file_path: str) -> list[dict[str, Any]]:
        if not smart_exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")

        requests: list[dict[str, Any]] = []
        with smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw_req = json.loads(line)
                    request_body, metadata = split_dataset_request(raw_req)
                    prepared_req = self.prepare_request(request_body)
                    requests.append(
                        {
                            "data_index": line_num,
                            "raw": request_body,
                            "prepared": prepared_req,
                            "hash": compute_hash(prepared_req),
                            "meta": metadata,
                        }
                    )
                except json.JSONDecodeError as e:
                    logger.error("JSON parse error at line %s: %s", line_num, e)
                except Exception as e:
                    logger.error("Error processing line %s: %s", line_num, e)

        logger.info("Successfully read %s requests", len(requests))
        return requests

    def build_request_record(
        self,
        request: Mapping[str, Any],
        *,
        data_index: int = 1,
    ) -> dict[str, Any]:
        request_body, metadata = split_dataset_request(request)
        prepared_req = self.prepare_request(request_body)
        return {
            "data_index": data_index,
            "raw": request_body,
            "prepared": prepared_req,
            "hash": compute_hash(prepared_req),
            "meta": metadata,
        }

    def read_result_jsonl(self, file_path: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with smart_open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error("Parse error at line %s in result file: %s", line_num, e)
        return results

    async def send_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        attempt = 0
        while True:
            try:
                async with self.semaphore:
                    return await self._send_once(request)
            except Exception as e:
                if not _is_retryable_exception(e):
                    logger.error("Request failed: %s", e)
                    return "failed", _serialize_error(e)

                delay = _compute_backoff_delay(attempt)
                attempt += 1
                logger.warning(
                    "Retryable error (%s), attempt %s, retrying in %.1fs: %s",
                    type(e).__name__,
                    attempt,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

    async def _send_once(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if request.get("stream", False):
            return await self._handle_stream_request(request)

        if not self.use_raw_completions:
            response = await self.client.chat.completions.create(**request, extra_body=self.extra_body)
        else:
            response = await self.client.completions.create(**request, extra_body=self.extra_body)

        return "success", response.model_dump()

    async def _handle_stream_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not self.use_raw_completions:
            stream = await self.client.chat.completions.create(**request, extra_body=self.extra_body)
        else:
            stream = await self.client.completions.create(**request, extra_body=self.extra_body)

        request_id = None
        created = None
        full_content: list[str] = []
        full_reasoning_content: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage = None

        async for event in stream:
            if getattr(event, "id", None):
                request_id = event.id
            if getattr(event, "created", None):
                created = event.created

            if not getattr(event, "choices", None):
                logger.warning("Empty choices in stream event")
                continue

            choice = event.choices[0]

            if getattr(choice, "delta", None):
                delta = choice.delta
                if getattr(delta, "content", None):
                    full_content.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    full_reasoning_content.append(delta.reasoning_content)
                if getattr(delta, "tool_calls", None):
                    self._accumulate_tool_calls(delta.tool_calls, tool_calls)
            elif getattr(choice, "text", None):
                full_content.append(choice.text)

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            if getattr(choice, "usage", None):
                usage = choice.usage

        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump()

        content_text = "".join(full_content)
        reasoning_content_text = "".join(full_reasoning_content) if full_reasoning_content else None
        if self.use_raw_completions:
            extracted_tool_calls = extract_tool_call_info(content_text)
            if extracted_tool_calls:
                tool_calls = {i: tc for i, tc in enumerate(extracted_tool_calls)}
                finish_reason = "tool_calls"

        tool_calls_list = list(tool_calls.values()) if tool_calls else None
        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "tool_calls": tool_calls_list,
        }
        if reasoning_content_text:
            message_dict["reasoning_content"] = reasoning_content_text

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage,
        }
        return "success", response

    def _accumulate_tool_calls(
        self,
        delta_tool_calls: list[Any],
        tool_calls: dict[int, dict[str, Any]],
    ) -> None:
        for tc in delta_tool_calls:
            raw_index = _read_attr_or_key(tc, "index")
            idx = raw_index if isinstance(raw_index, int) else 0

            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": _read_attr_or_key(tc, "id"),
                    "type": _read_attr_or_key(tc, "type") or "function",
                    "function": {"name": "", "arguments": ""},
                }

            function = _read_attr_or_key(tc, "function")
            if function:
                name = _read_attr_or_key(function, "name")
                if name:
                    tool_calls[idx]["function"]["name"] = name
                arguments = _read_attr_or_key(function, "arguments")
                if arguments:
                    tool_calls[idx]["function"]["arguments"] += arguments

    async def process_request(self, prepared_req: dict[str, Any], data_index: int) -> dict[str, Any]:
        start_time = time.time()
        status, response = await self.send_request(prepared_req["prepared"])
        duration_ms = int((time.time() - start_time) * 1000)

        finish_reason = None
        tool_calls_present = False
        tool_calls_valid = None
        tool_call_names: list[str] = []
        expected_tool_call_names: list[str] = []
        tool_call_names_match = None

        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            tools = prepared_req["raw"].get("tools", [])
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            if tool_calls:
                tool_calls_present = True
                tool_calls_valid = all(self.validate_tool_call(tc, tools) for tc in tool_calls)
                tool_call_names = [
                    str(tool_call.get("function", {}).get("name") or "")
                    for tool_call in tool_calls
                    if isinstance(tool_call, Mapping)
                ]
            expected_tool_call_names = [
                str(name).strip()
                for name in prepared_req.get("meta", {}).get("expected_tool_call_names", [])
                if str(name).strip()
            ]
            if expected_tool_call_names:
                tool_call_names_match = Counter(tool_call_names) == Counter(expected_tool_call_names)

        result = {
            "data_index": data_index,
            "case_id": prepared_req.get("meta", {}).get("case_id"),
            "case_description": prepared_req.get("meta", {}).get("description"),
            "request": prepared_req["prepared"],
            "extra_body": self.extra_body,
            "response": response,
            "status": status,
            "finish_reason": finish_reason,
            "tool_calls_present": tool_calls_present,
            "tool_calls_valid": tool_calls_valid,
            "tool_call_names": tool_call_names,
            "expected_tool_call_names": expected_tool_call_names,
            "tool_call_names_match": tool_call_names_match,
            "last_run_at": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "hash": prepared_req["hash"],
        }
        return result

    async def validate_request(
        self,
        request: Mapping[str, Any],
        *,
        data_index: int = 1,
    ) -> dict[str, Any]:
        request_record = self.build_request_record(request, data_index=data_index)
        return await self.process_request(request_record, data_index)

    def validate_tool_call(self, tool_call: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
        return validate_tool_call_against_tools(tool_call, tools)

    async def validate_file(self, file_path: str) -> None:
        self.eval_start_ts = time.time()
        self.eval_end_ts = None
        self.eval_started_at = datetime.now().isoformat()
        self.eval_finished_at = None

        all_requests = self.read_jsonl(file_path)
        if not all_requests:
            logger.warning("Test set is empty, no requests to process")
            return

        existing_hash_map: dict[str, dict[str, Any]] = {}

        if self.incremental and smart_exists(self.output_file):
            existing_results = self.read_result_jsonl(self.output_file)
            for result in existing_results:
                existing_hash_map[result["hash"]] = result
            logger.info("Incremental mode: loaded %s existing results", len(existing_results))
        else:
            async with self.file_lock:
                with smart_open(self.output_file, "w", encoding="utf-8") as f:
                    pass
            logger.info("Initialized output file: %s", self.output_file)

        await self.update_summary_file()

        tasks: list[asyncio.Task[dict[str, Any]]] = []
        self.results = []

        for req in all_requests:
            request_hash = req["hash"]
            data_index = req["data_index"]

            if self.incremental and request_hash in existing_hash_map:
                existing = existing_hash_map[request_hash]
                if existing.get("status") == "success":
                    self.results.append(existing)
                    continue

            tasks.append(asyncio.create_task(self.process_request(req, data_index)))

        if not tasks:
            logger.info("All requests already processed successfully, no need to rerun")
            return

        logger.info("Preparing to process %s requests", len(tasks))

        with _progress(total=len(tasks), desc="Processing", unit="req") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    res = await task
                    finish_reason = res.get("finish_reason")
                    self.finish_reason_stat[finish_reason] = self.finish_reason_stat.get(finish_reason, 0) + 1
                    self.results.append(res)
                    await self.save_result_and_update_stats(res)
                except Exception as e:
                    logger.error("Task execution failed: %s", e)
                finally:
                    pbar.update(1)

        await self.deduplicate_and_sort_results()

        self.eval_end_ts = time.time()
        self.eval_finished_at = datetime.now().isoformat()

        await self.update_summary_file()

        logger.info("Results saved to: %s", self.output_file)
        logger.info("Summary saved to: %s", self.summary_file)

    async def save_result_and_update_stats(self, result: dict[str, Any]) -> None:
        async with self.file_lock:
            with smart_open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        async with self.stats_lock:
            summary = self.compute_summary()
            await self.update_summary_file(summary)
            logger.info(
                "[Stats] Total: %s, Success: %s, Failed: %s, Stop: %s, ToolCalls: %s, ToolCallValid: %s, ToolCallInvalid: %s",
                summary["success_count"] + summary["failure_count"],
                summary["success_count"],
                summary["failure_count"],
                summary["finish_stop"],
                summary["finish_tool_calls"],
                summary["successful_tool_call_count"],
                summary["schema_validation_error_count"],
            )

    async def deduplicate_and_sort_results(self) -> None:
        if not smart_exists(self.output_file):
            logger.warning("Output file does not exist: %s", self.output_file)
            return

        all_results = self.read_result_jsonl(self.output_file)
        if not all_results:
            logger.info("No results to process")
            return

        logger.info("Processing %s results for deduplication and sorting", len(all_results))

        results_by_index: dict[int, dict[str, Any]] = {}
        for result in all_results:
            data_index = result.get("data_index")
            if data_index is None:
                logger.warning("Result missing data_index: %s", result)
                continue

            last_run_at = result.get("last_run_at")
            if last_run_at is None:
                logger.warning("Result missing last_run_at: %s", result)
                continue

            if data_index not in results_by_index:
                results_by_index[data_index] = result
            else:
                existing_last_run = results_by_index[data_index].get("last_run_at")
                if existing_last_run is None or last_run_at > existing_last_run:
                    results_by_index[data_index] = result

        deduplicated_results = list(results_by_index.values())
        deduplicated_results.sort(key=lambda x: x.get("data_index", 0))

        logger.info("Deduplicated from %s to %s results", len(all_results), len(deduplicated_results))

        async with self.file_lock:
            with smart_open(self.output_file, "w", encoding="utf-8") as f:
                for result in deduplicated_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        self.results = deduplicated_results
        logger.info("Results deduplicated, sorted, and saved to: %s", self.output_file)

    async def update_summary_file(self, summary: dict[str, Any] | None = None) -> None:
        if summary is None:
            summary = self.compute_summary()
        with smart_open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

    def compute_summary(self) -> dict[str, Any]:
        eval_duration_ms = None
        if isinstance(self.eval_start_ts, (int, float)):
            end_ts = self.eval_end_ts if isinstance(self.eval_end_ts, (int, float)) else time.time()
            eval_duration_ms = int(max(0.0, (end_ts - self.eval_start_ts) * 1000))

        self.summary = build_summary(
            self.results,
            model=self.model,
            eval_started_at=self.eval_started_at,
            eval_finished_at=self.eval_finished_at,
            eval_duration_ms=eval_duration_ms,
        )
        return self.summary


def prepare_request_payload(
    request: Mapping[str, Any],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra_body: Mapping[str, Any] | None = None,
    normalize_tool_call_ids: bool = True,
    use_raw_completions: bool = False,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    request_body, _ = split_dataset_request(request)
    req = (
        normalize_historical_tool_call_ids(request_body)
        if normalize_tool_call_ids
        else copy.deepcopy(request_body)
    )

    if "messages" in req:
        for message in req["messages"]:
            if message.get("role") == ROLE_INPUT:
                message["role"] = ROLE_SYSTEM

    req["model"] = model

    if temperature is not None:
        req["temperature"] = temperature
    if max_tokens is not None:
        if "max_completion_tokens" in req and "max_tokens" not in req:
            req["max_completion_tokens"] = max_tokens
        else:
            req["max_tokens"] = max_tokens

    if req.get("stream", False) and not use_raw_completions:
        so = req.get("stream_options")
        if not isinstance(so, dict):
            so = {}
        so.setdefault("include_usage", True)
        req["stream_options"] = so

    if use_raw_completions:
        if tokenizer is None:
            raise RuntimeError(
                "Raw completions mode requires `transformers`. Install it and rerun with `--tokenizer-model` if needed."
            )
        if "max_completion_tokens" in req and "max_tokens" not in req:
            req["max_tokens"] = req.pop("max_completion_tokens")
        req["prompt"] = tokenizer.apply_chat_template(
            req["messages"],
            tokenize=False,
            tools=req.get("tools", None),
            add_generation_prompt=True,
        )
        req.pop("messages")
        if "tools" in req:
            req.pop("tools")

    if extra_body:
        req.setdefault("_extra_body_preview", dict(extra_body))

    return req


async def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        extra_body = parse_extra_body(args.extra_body)
        output_file, summary_file = build_default_report_paths(args.output, args.summary)
    except ValueError as e:
        parser.error(str(e))

    async with ToolCallsValidator(
        model=args.model,
        base_url=resolve_cli_base_url(args.base_url),
        api_key=args.api_key,
        concurrency=args.concurrency,
        output_file=output_file,
        summary_file=summary_file,
        timeout=args.timeout,
        max_retries=args.retries,
        extra_body=extra_body,
        incremental=args.incremental,
        use_raw_completions=args.use_raw_completions,
        tokenizer_model=args.tokenizer_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        normalize_tool_call_ids=not args.disable_tool_call_id_normalization,
    ) as validator:
        await validator.validate_file(args.file_path)
    return 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main(argv))
