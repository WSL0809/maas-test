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

## B6 工具调用-并行调用

- 对应用例：
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[glm5-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[minimax-m25-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[kimi-k25-parallel_distinct_tool_calls]`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm5 --chat-model minimax-m25 --chat-model kimi-k25
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm5`、`minimax-m25`、`kimi-k25` 可稳定返回同一条 assistant 消息中的两个不同 tool calls。
  - `qwen35` 当前会耗尽 completion 长度，只输出 reasoning，不产出结构化 `tool_calls`。
  - `minimax-m21` 当前会以文本/XML 形式写出工具调用并因长度截断，未进入结构化 `tool_calls` 通道。

## B7 工具调用-多步链式

- 对应用例：
  - `tests/test_tool_calling.py::test_multi_step_tool_chain_round_trip`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm5 --chat-model minimax-m25 --chat-model minimax-m21 -rx
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm5`、`minimax-m21`、`minimax-m25` 可稳定完成 `fetch_seed_word -> uppercase_word -> decorate_word -> [STONE]` 的 3 步链式回填。
  - `qwen35` 首步可进入工具调用，但第二步会退化为文本/XML 风格的伪工具调用并因长度截断，未进入结构化 `tool_calls`。
  - `kimi-k25` 当前可完成前两步，但第三步会丢失结构化 `tool_calls`，只剩 reasoning，导致链路无法闭环。

## B8 JSON Mode

- 对应用例：
  - `tests/test_chat.py::test_json_mode_returns_valid_json_object`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `qwen35`、`kimi-k25` 可在 `message.content` 稳定返回合法 JSON。
  - `glm5`、`minimax-m25`、`minimax-m21` 当前会把 JSON 放在 `message.reasoning` 且 `message.content=null`，在该用例中以 `xfail` 记录。

## B9 结构化输出

- 对应用例：
  - `tests/test_chat.py::test_structured_output_tool_returns_valid_arguments`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm5`、`qwen35`、`minimax-m25`、`minimax-m21`、`kimi-k25` 均可稳定通过。
