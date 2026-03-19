#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "aiohttp>=3.10",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "glm-5"
DEFAULT_BASE_URL = "https://api.openai.com"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"


def parse_env_file(path: Path) -> dict[str, str]:
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
        os.environ.setdefault(key, value)


def normalize_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized or DEFAULT_BASE_URL


def resolve_base_url(url_override: str | None) -> str:
    if url_override:
        return normalize_base_url(url_override)
    env_url = os.getenv("OPENAI_BASE_URL")
    if env_url:
        return normalize_base_url(env_url)
    return DEFAULT_BASE_URL


def get_api_key(api_key_override: str | None) -> str:
    if api_key_override is not None:
        return api_key_override
    return os.getenv("OPENAI_API_KEY", "")


def build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def parse_extra_body(raw_json: str | None) -> dict[str, Any]:
    if raw_json is None:
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json must decode to a JSON object.")
    return parsed


def deep_merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def non_empty_string(value: Any) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def first_choice(chunk: dict[str, Any]) -> dict[str, Any]:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def summarize_stream_chunks(chunks: list[dict[str, Any]], saw_done: bool) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    content_parts: list[str] = []
    text_parts: list[str] = []
    content_channel_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_content_parts: list[str] = []
    combined_reasoning_parts: list[str] = []
    finish_reasons: list[str] = []
    usage_chunk_indexes: list[int] = []

    first_delta_content_ms: float | None = None
    first_content_channel_ms: float | None = None
    first_reasoning_ms: float | None = None
    first_reasoning_field: str | None = None

    for index, chunk in enumerate(chunks, start=1):
        choice = first_choice(chunk)
        delta = choice.get("delta")
        delta_map = delta if isinstance(delta, dict) else {}
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reasons.append(finish_reason)

        content = non_empty_string(delta_map.get("content"))
        text = non_empty_string(delta_map.get("text"))
        reasoning = non_empty_string(delta_map.get("reasoning"))
        reasoning_content = non_empty_string(delta_map.get("reasoning_content"))
        relative_ms = chunk.get("_received_at_ms")
        usage = chunk.get("usage")
        has_usage = usage is not None
        if has_usage:
            usage_chunk_indexes.append(index)

        if content is not None:
            content_parts.append(content)
            content_channel_parts.append(content)
            if first_delta_content_ms is None and isinstance(relative_ms, (int, float)):
                first_delta_content_ms = float(relative_ms)

        if text is not None:
            text_parts.append(text)
            if content is None:
                content_channel_parts.append(text)

        if first_content_channel_ms is None and (content is not None or text is not None):
            if isinstance(relative_ms, (int, float)):
                first_content_channel_ms = float(relative_ms)

        if reasoning is not None:
            reasoning_parts.append(reasoning)
            combined_reasoning_parts.append(reasoning)
            if first_reasoning_ms is None and isinstance(relative_ms, (int, float)):
                first_reasoning_ms = float(relative_ms)
                first_reasoning_field = "delta.reasoning"

        if reasoning_content is not None:
            reasoning_content_parts.append(reasoning_content)
            combined_reasoning_parts.append(reasoning_content)
            if first_reasoning_ms is None and isinstance(relative_ms, (int, float)):
                first_reasoning_ms = float(relative_ms)
                first_reasoning_field = "delta.reasoning_content"

        observations.append(
            {
                "index": index,
                "received_at_ms": relative_ms,
                "delta_keys": sorted(delta_map.keys()),
                "delta_content": content,
                "delta_text": text,
                "delta_reasoning": reasoning,
                "delta_reasoning_content": reasoning_content,
                "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
                "usage_present": has_usage,
                "usage": usage if isinstance(usage, dict) else usage,
            }
        )

    has_non_empty_delta_content = len(content_parts) > 0
    has_non_empty_content_channel = len(content_channel_parts) > 0
    has_non_empty_reasoning = len(combined_reasoning_parts) > 0

    field_order: str
    if first_reasoning_ms is None and first_delta_content_ms is None:
        field_order = "none"
    elif first_reasoning_ms is not None and first_delta_content_ms is None:
        field_order = "reasoning_only"
    elif first_reasoning_ms is None and first_delta_content_ms is not None:
        field_order = "content_only"
    elif first_reasoning_ms < first_delta_content_ms:
        field_order = "reasoning_before_content"
    elif first_reasoning_ms > first_delta_content_ms:
        field_order = "content_before_reasoning"
    else:
        field_order = "same_time"

    usage_only_on_terminal_chunk = False
    if usage_chunk_indexes:
        usage_only_on_terminal_chunk = all(index == len(chunks) for index in usage_chunk_indexes)

    return {
        "saw_done": saw_done,
        "chunk_count": len(chunks),
        "chunk_observations": observations,
        "finish_reasons": finish_reasons,
        "aggregated_content": "".join(content_channel_parts),
        "aggregated_delta_content": "".join(content_parts),
        "aggregated_delta_text": "".join(text_parts),
        "aggregated_reasoning": "".join(reasoning_parts),
        "aggregated_reasoning_content": "".join(reasoning_content_parts),
        "aggregated_combined_reasoning": "".join(combined_reasoning_parts),
        "has_non_empty_delta_content": has_non_empty_delta_content,
        "has_non_empty_content_channel": has_non_empty_content_channel,
        "has_non_empty_reasoning": has_non_empty_reasoning,
        "first_delta_content_at_ms": first_delta_content_ms,
        "first_content_channel_at_ms": first_content_channel_ms,
        "first_reasoning_at_ms": first_reasoning_ms,
        "first_reasoning_field": first_reasoning_field,
        "first_field_order": field_order,
        "reasoning_only_stream": has_non_empty_reasoning and not has_non_empty_delta_content,
        "reasoning_before_content": (
            first_reasoning_ms is not None
            and first_delta_content_ms is not None
            and first_reasoning_ms < first_delta_content_ms
        ),
        "content_never_appears": not has_non_empty_delta_content,
        "usage_chunk_indexes": usage_chunk_indexes,
        "usage_only_on_terminal_chunk": usage_only_on_terminal_chunk,
    }


def summarize_nonstream_response(response_json: dict[str, Any]) -> dict[str, Any]:
    choice = first_choice(response_json)
    message = choice.get("message")
    message_map = message if isinstance(message, dict) else {}
    content = message_map.get("content")
    reasoning = message_map.get("reasoning")
    reasoning_content = message_map.get("reasoning_content")
    return {
        "finish_reason": choice.get("finish_reason")
        if isinstance(choice.get("finish_reason"), str)
        else None,
        "content": content if isinstance(content, str) else None,
        "reasoning": reasoning if isinstance(reasoning, str) else None,
        "reasoning_content": reasoning_content if isinstance(reasoning_content, str) else None,
        "usage": response_json.get("usage") if isinstance(response_json.get("usage"), dict) else None,
    }


def compare_stream_and_nonstream(
    stream_summary: dict[str, Any], nonstream_summary: dict[str, Any]
) -> dict[str, Any]:
    stream_content = stream_summary["aggregated_content"]
    stream_reasoning = stream_summary["aggregated_reasoning"]
    stream_reasoning_content = stream_summary["aggregated_reasoning_content"]
    stream_combined_reasoning = stream_summary["aggregated_combined_reasoning"]

    nonstream_content = nonstream_summary["content"] or ""
    nonstream_reasoning = nonstream_summary["reasoning"] or ""
    nonstream_reasoning_content = nonstream_summary["reasoning_content"] or ""
    nonstream_combined_reasoning = nonstream_reasoning + nonstream_reasoning_content

    return {
        "content_match": stream_content == nonstream_content,
        "reasoning_match": stream_reasoning == nonstream_reasoning,
        "reasoning_content_match": stream_reasoning_content == nonstream_reasoning_content,
        "combined_reasoning_match": stream_combined_reasoning == nonstream_combined_reasoning,
        "stream_content_length": len(stream_content),
        "stream_reasoning_length": len(stream_combined_reasoning),
        "nonstream_content_length": len(nonstream_content),
        "nonstream_reasoning_length": len(nonstream_combined_reasoning),
    }


def format_payload(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    base_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    return deep_merge_dict(base_payload, extra_body)


async def run_stream_request(
    session: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_timeout_sec: int,
) -> tuple[list[dict[str, Any]], bool]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=request_timeout_sec)
    chunks: list[dict[str, Any]] = []
    saw_done = False
    start_ns = time.perf_counter_ns()
    async with session.post(url=url, json=payload, headers=headers, timeout=timeout) as response:
        body = await response.text() if response.status != 200 else None
        if response.status != 200:
            raise RuntimeError(f"Stream request failed: HTTP {response.status} body={body}")

        async for line_bytes in response.content:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if payload_text == "[DONE]":
                saw_done = True
                continue
            chunk = json.loads(payload_text)
            chunk["_received_at_ms"] = round((time.perf_counter_ns() - start_ns) / 1_000_000, 3)
            chunks.append(chunk)
    return chunks, saw_done


async def run_nonstream_request(
    session: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_timeout_sec: int,
) -> dict[str, Any]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=request_timeout_sec)
    async with session.post(url=url, json=payload, headers=headers, timeout=timeout) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"Non-stream request failed: HTTP {response.status} body={body}")
        return json.loads(body)


def build_messages(system_prompt: str | None, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def print_summary(
    endpoint: str,
    stream_summary: dict[str, Any],
    nonstream_summary: dict[str, Any],
    comparison: dict[str, Any],
    verbose_raw: bool,
) -> None:
    print("=" * 80)
    print(f"Endpoint: {endpoint}")
    print(f"Stream chunks: {stream_summary['chunk_count']} | saw [DONE]: {stream_summary['saw_done']}")
    print(f"Has non-empty delta.content: {stream_summary['has_non_empty_delta_content']}")
    print(f"Has non-empty content channel (delta.content or delta.text): {stream_summary['has_non_empty_content_channel']}")
    print(f"Has non-empty reasoning: {stream_summary['has_non_empty_reasoning']}")
    print(f"First field order (reasoning vs delta.content): {stream_summary['first_field_order']}")
    print(f"Reasoning first timestamp (ms): {stream_summary['first_reasoning_at_ms']}")
    print(f"Content first timestamp (ms): {stream_summary['first_delta_content_at_ms']}")
    print(f"Reasoning first field: {stream_summary['first_reasoning_field']}")
    print(f"reasoning_only_stream: {stream_summary['reasoning_only_stream']}")
    print(f"reasoning_before_content: {stream_summary['reasoning_before_content']}")
    print(f"content_never_appears: {stream_summary['content_never_appears']}")
    print(f"usage_only_on_terminal_chunk: {stream_summary['usage_only_on_terminal_chunk']}")
    print("-" * 80)
    print(f"Aggregated content length: {len(stream_summary['aggregated_content'])}")
    print(f"Aggregated reasoning length: {len(stream_summary['aggregated_combined_reasoning'])}")
    print(f"Stream finish reasons: {stream_summary['finish_reasons']}")
    print("-" * 80)
    print(f"Non-stream content length: {len(nonstream_summary['content'] or '')}")
    print(
        "Non-stream reasoning length: "
        f"{len((nonstream_summary['reasoning'] or '') + (nonstream_summary['reasoning_content'] or ''))}"
    )
    print(f"Non-stream finish_reason: {nonstream_summary['finish_reason']}")
    print("-" * 80)
    print(f"content_match: {comparison['content_match']}")
    print(f"reasoning_match: {comparison['reasoning_match']}")
    print(f"reasoning_content_match: {comparison['reasoning_content_match']}")
    print(f"combined_reasoning_match: {comparison['combined_reasoning_match']}")
    print("=" * 80)

    if verbose_raw:
        print("Raw stream chunk observations:")
        print(json.dumps(stream_summary["chunk_observations"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe stream vs non-stream channel behavior for chat completions."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to probe.")
    parser.add_argument("--url", default=None, help="Base URL. Accepts with or without /v1.")
    parser.add_argument("--api-key", default=None, help="API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument(
        "--prompt",
        default="Please solve 6*7 and explain your steps briefly, then give a final short answer.",
        help="User prompt for a single-turn probe.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a precise assistant.",
        help="Optional system prompt. Use empty string to omit.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="max_completion_tokens value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into request body.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write probe artifact JSON.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--verbose-raw",
        action="store_true",
        help="Print per-chunk raw observation details.",
    )
    return parser.parse_args()


async def main() -> None:
    import aiohttp

    load_dotenv()
    args = parse_args()
    base_url = resolve_base_url(args.url)
    endpoint = f"{base_url}{CHAT_COMPLETIONS_PATH}"
    api_key = get_api_key(args.api_key)
    headers = build_headers(api_key)
    extra_body = parse_extra_body(args.extra_body_json)

    system_prompt = args.system_prompt if args.system_prompt else None
    messages = build_messages(system_prompt=system_prompt, prompt=args.prompt)
    base_payload = format_payload(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        extra_body=extra_body,
    )

    stream_payload = deep_merge_dict(
        base_payload,
        {
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    nonstream_payload = deep_merge_dict(base_payload, {"stream": False})

    async with aiohttp.ClientSession() as session:
        stream_chunks, saw_done = await run_stream_request(
            session=session,
            url=endpoint,
            payload=stream_payload,
            headers=headers,
            request_timeout_sec=args.request_timeout_sec,
        )
        nonstream_response = await run_nonstream_request(
            session=session,
            url=endpoint,
            payload=nonstream_payload,
            headers=headers,
            request_timeout_sec=args.request_timeout_sec,
        )

    stream_summary = summarize_stream_chunks(stream_chunks, saw_done=saw_done)
    nonstream_summary = summarize_nonstream_response(nonstream_response)
    comparison = compare_stream_and_nonstream(stream_summary, nonstream_summary)

    print_summary(
        endpoint=endpoint,
        stream_summary=stream_summary,
        nonstream_summary=nonstream_summary,
        comparison=comparison,
        verbose_raw=args.verbose_raw,
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "generated_at": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "request": {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "max_completion_tokens": args.max_completion_tokens,
                "extra_body": extra_body,
                "stream_payload": stream_payload,
                "nonstream_payload": nonstream_payload,
            },
            "stream": {
                "raw_chunks": stream_chunks,
                "summary": stream_summary,
            },
            "non_stream": {
                "response": nonstream_response,
                "summary": nonstream_summary,
            },
            "comparison": comparison,
        }
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote artifact: {output_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
