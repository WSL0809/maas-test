# Test Run Commands

## B2 非思考模式（Instant）

- B2-A 请求可接受性
- 对应用例：
  - `tests/test_chat.py::test_create_accepts_chat_template_kwargs_enable_thinking_false`
  - `tests/test_chat.py::test_stream_accepts_chat_template_kwargs_enable_thinking_false`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_accepts_chat_template_kwargs_enable_thinking_false or test_stream_accepts_chat_template_kwargs_enable_thinking_false'
```

- B2-B 严格 suppress reasoning
- 对应用例：
  - `tests/test_chat.py::test_create_suppresses_reasoning_when_thinking_disabled`
  - `tests/test_chat.py::test_stream_suppresses_reasoning_when_thinking_disabled`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_suppresses_reasoning_when_thinking_disabled or test_stream_suppresses_reasoning_when_thinking_disabled' -rxX
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_accepts_chat_template_kwargs_enable_thinking_false or test_stream_accepts_chat_template_kwargs_enable_thinking_false'
```

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_suppresses_reasoning_when_thinking_disabled or test_stream_suppresses_reasoning_when_thinking_disabled' -rxX
```
