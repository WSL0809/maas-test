from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODULE_PATH = REPO_ROOT / "multi_turn" / "benchmark_serving_multi_turn.py"
SPEC = importlib.util.spec_from_file_location("benchmark_serving_multi_turn", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None

fake_aiohttp = types.ModuleType("aiohttp")
fake_aiohttp.ClientSession = object
fake_aiohttp.ClientTimeout = lambda total: {"total": total}
sys.modules.setdefault("aiohttp", fake_aiohttp)

fake_numpy = types.ModuleType("numpy")
fake_numpy.ndarray = object
fake_numpy.random = types.SimpleNamespace(
    randint=lambda *args, **kwargs: [],
    uniform=lambda *args, **kwargs: [],
    zipf=lambda *args, **kwargs: [],
    poisson=lambda *args, **kwargs: [],
    lognormal=lambda *args, **kwargs: [],
)
fake_numpy.minimum = lambda values, max_val: values
fake_numpy.full = lambda shape, fill_value: [fill_value] * shape
fake_numpy.sqrt = lambda value: value
fake_numpy.log = lambda value: value
fake_numpy.round = lambda value: value
sys.modules.setdefault("numpy", fake_numpy)

sys.modules.setdefault("pandas", types.ModuleType("pandas"))

fake_tqdm = types.ModuleType("tqdm")
fake_tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", fake_tqdm)

fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoTokenizer = object
sys.modules.setdefault("transformers", fake_transformers)

benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_parse_extra_body_requires_object_json() -> None:
    assert benchmark.parse_extra_body('{"chat_template_kwargs":{"enable_thinking":false}}') == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    with pytest.raises(ValueError, match="JSON object"):
        benchmark.parse_extra_body('["not", "an", "object"]')


def test_deep_merge_dict_merges_nested_request_fields() -> None:
    base = {
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True, "foo": "base"},
    }
    extra = {
        "chat_template_kwargs": {"enable_thinking": False, "bar": "extra"},
        "top_p": 0.8,
    }

    assert benchmark.deep_merge_dict(base, extra) == {
        "temperature": 0.0,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "foo": "base",
            "bar": "extra",
        },
        "top_p": 0.8,
    }


class FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeResponse:
    def __init__(self, *, status: int, chunks: list[bytes]) -> None:
        self.status = status
        self.content = FakeContent(chunks)

    async def text(self) -> str:
        return ""


class FakeRequestContext:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.last_json = None

    def post(self, *, url, json, headers, timeout):
        self.last_json = json
        return FakeRequestContext(self._response)


def test_send_request_merges_extra_body_into_payload() -> None:
    chunk = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }
    ).encode("utf-8")
    session = FakeSession(FakeResponse(status=200, chunks=[chunk]))

    response = asyncio.run(
        benchmark.send_request(
            session=session,
            messages=[{"role": "user", "content": "hi"}],
            chat_url="http://example.test/v1/chat/completions",
            model="demo-model",
            stream=False,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )

    assert response.valid is True
    assert response.content == "ok"
    assert session.last_json == {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hi"}],
        "seed": 0,
        "temperature": 0.0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
