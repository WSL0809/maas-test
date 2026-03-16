# Repository Guidelines

## Project Structure & Module Organization

This repository is intentionally small and centered on live API validation.

- [test_chat.py](/Users/wangshilong/Downloads/maas-test/test_chat.py): the base `httpx` chat suite for create, stream, and StructuredOutput coverage.
- [test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/test_tool_calling.py): the `httpx` tool-calling suite plus optional SDK tool-calling probe coverage.
- [test_chat_sdk_smoke.py](/Users/wangshilong/Downloads/maas-test/test_chat_sdk_smoke.py): small OpenAI Python SDK smoke suite. Default skip; enable explicitly.
- [chat_test_support.py](/Users/wangshilong/Downloads/maas-test/chat_test_support.py): shared client, request, SSE parsing, and failure-artifact helpers.
- [pyproject.toml](/Users/wangshilong/Downloads/maas-test/pyproject.toml): Python project metadata and runtime/test dependencies.
- [uv.lock](/Users/wangshilong/Downloads/maas-test/uv.lock): locked dependency graph for reproducible installs.
- [api.md](/Users/wangshilong/Downloads/maas-test/api.md): local API reference snapshot used to scope test coverage.
- [third_party/vllm](/Users/wangshilong/Downloads/maas-test/third_party/vllm): upstream vLLM source snapshot kept as a git submodule for backend behavior and compatibility reference.
- [third_party/opencode](/Users/wangshilong/Downloads/maas-test/third_party/opencode): upstream OpenCode source kept as a git submodule for external implementation reference.
- `.env`: local credentials and endpoint settings for live runs. Keep it local only.
- `test_failure_artifacts/`: generated only when a test fails; stores full request/response artifacts for debugging.
- test_chat.py 中的每个测试函数，都要在 README 有对应的解释
- 当你想使用 python 执行命令时，请申请提权后使用 uv run python
- 本项目测试的后端服务是由 vllm 提供的openai api 兼容的推理服务
- 外部源码目录统一使用 `third_party/`，不要新建 `third-party/` 这种变体目录

## Build, Test, and Development Commands

- `uv sync`: install and lock the project environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest -q test_chat.py test_tool_calling.py`: run the default per-model optimized live suite.
- `uv run pytest -q test_chat.py test_tool_calling.py --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25`: run the optimized suite against a chosen model matrix.
- `uv run pytest -q test_tool_calling.py --run-tool-calling-probe`: run the SDK tool-calling probe coverage.
- `uv run pytest -q test_chat_sdk_smoke.py --run-sdk-smoke`: run the OpenAI SDK smoke suite.
- `uv run pytest -q test_chat.py test_tool_calling.py test_chat_sdk_smoke.py --run-sdk-smoke`: run the default suite plus SDK smoke.

## Coding Style & Naming Conventions

- Use Python 3.12 features already present in the repo.
- Follow PEP 8 with 4-space indentation and clear, compact helper functions.
- Name tests as `test_<behavior>` and keep assertions behavior-focused rather than tied to exact model wording.
- Prefer small helper functions for repeated live-call setup, such as response normalization or reasoning extraction.

## Testing Guidelines

- Tests use `pytest` and call a live OpenAI-compatible endpoint.
- Keep prompts deterministic with `temperature=0`; use fixed seeds when comparing reasoning behavior.
- The default suite is allowed to encode model-specific passing paths through per-model subclasses instead of forcing a single universal request shape.
- Assert stable signals such as response shape, tool call arguments, stream events, and structured-output channel placement. Avoid brittle full-text assertions.
- Run tests from the repo root so `.env` is loaded automatically by `test_chat.py`.

## Commit & Pull Request Guidelines

- Existing history uses short imperative messages such as `Create .gitignore` and `init gitignore file content`.
- Prefer concise commit titles in imperative mood, for example: `Add reasoning coverage for chat tests`.
- PRs should describe which API behaviors changed, which command was run for verification, and whether the change depends on a specific endpoint/model combination.

## Security & Configuration Tips

- Do not commit `.env`; `.gitignore` already excludes it.
- Treat API keys and custom base URLs as secrets.
- If tests fail unexpectedly, first confirm `.env` values, model availability, and endpoint compatibility with chat completions features.

注意事项：
1. 在执行脚本的时候尽量启动一个 sub-agent 执行，返回执行结果即可
