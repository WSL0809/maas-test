# Test Run Commands

`main.py` 会把这个文件当作机器可读的命令源来解析。维护约定如下：

- 每个 `## <CASE_ID>` 小节都应保留一个明确标注的默认命令。
- 标签里包含 `默认` 的命令会被 `main.py` 优先选择。
- 如果同一小节出现多个 `默认...命令`，`main.py` 当前会只取第一个，并在生成报告时写入告警。
- `单模型复测示例` 默认只用于人工复查；只有当该小节没有其他可运行命令时，`main.py` 才会退回使用它。

## H1 Chat Completions 兼容

- 对应用例：
  - `tests/test_api_compatibility.py::test_chat_completions_returns_openai_shape_and_usage`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_chat_completions_returns_openai_shape_and_usage --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m25`、`minimax-m21` 当前可返回 OpenAI-compatible 的 `chat.completion` 结构
  - `kimi-k25` 在当前环境可能直接命中上游 DNS/路由型 `500 do_request_failed`，导致 H1 在响应形状断言前失败

## H2 Raw Completions 兼容

- 对应用例：
  - `tests/test_api_compatibility.py::test_raw_completions_returns_openai_shape_and_usage`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_raw_completions_returns_openai_shape_and_usage --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m25`、`minimax-m21` 当前可返回 `text_completion` 结构，并带 `usage`
  - `kimi-k25` 在当前环境可能直接命中上游 DNS/路由型 `500 do_request_failed`，导致 H2 在响应形状断言前失败

## H3 /v1/models 模型列表

- 对应用例：
  - `tests/test_api_compatibility.py::test_models_endpoint_lists_selected_model`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_models_endpoint_lists_selected_model --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `/v1/models` 当前可返回 `list` 对象
  - 默认模型矩阵中的 5 个模型都能在返回列表里找到

## H4 Usage 统计

- 对应用例：
  - `tests/test_api_compatibility.py::test_chat_completions_returns_openai_shape_and_usage`
  - `tests/test_api_compatibility.py::test_raw_completions_returns_openai_shape_and_usage`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k 'test_chat_completions_returns_openai_shape_and_usage or test_raw_completions_returns_openai_shape_and_usage' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m25`、`minimax-m21` 当前在 `/v1/chat/completions` 与 `/v1/completions` 两条链路都满足 `prompt_tokens + completion_tokens == total_tokens`
  - `kimi-k25` 若继续返回上游 DNS/路由型 `500`，则该轮无法完成 H4 的 usage 算术校验

## H5 错误码规范

- 对应用例：
  - `tests/test_api_compatibility.py::test_chat_completions_bad_request_returns_openai_error_shape`
  - `tests/test_api_compatibility.py::test_chat_completions_invalid_api_key_returns_openai_error_shape`
  - `tests/test_api_compatibility.py::test_unknown_route_returns_openai_error_shape`
  - `tests/test_api_compatibility.py::TestGLM5RateLimitProbe::test_light_rate_limit_probe_returns_only_200_or_429`
  - `tests/test_api_compatibility.py::TestMinimaxM25ApiCompatibility::test_forced_named_tool_choice_returns_openai_error_shape_when_upstream_500_reproduces`
- 显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k 'test_chat_completions_bad_request_returns_openai_error_shape or test_chat_completions_invalid_api_key_returns_openai_error_shape or test_unknown_route_returns_openai_error_shape or test_light_rate_limit_probe_returns_only_200_or_429 or test_forced_named_tool_choice_returns_openai_error_shape_when_upstream_500_reproduces' --chat-model glm-5 --chat-model minimax-m25 -rxs
```

- 当前已知现象：
  - `400` 可通过 `messages` 类型错误稳定复现，错误体为 OpenAI-compatible 顶层 `error`
  - `401` 可通过无效 Bearer Token 稳定复现，错误体为 OpenAI-compatible 顶层 `error`
  - `404` 可通过不存在路径稳定复现，错误体为 OpenAI-compatible 顶层 `error`
  - `429` 采用轻量并发探测；当前环境可能全部返回 `200`，不保证每轮都触发限流
  - `500` 当前可通过 `minimax-m25` forced named `tool_choice` 路径探测；若后端已修复则该用例会 `skip`

## C1 单图理解

- 对应用例：
  - `tests/test_chat.py::test_create_understands_single_image_dominant_color`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_understands_single_image_dominant_color -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_understands_single_image_dominant_color --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_understands_single_image_dominant_color -rx
```

- 当前已知现象：
  - `qwen35`、`kimi-k25` 可稳定识别内置纯红色图片并返回 `red`
  - `glm-5`、`minimax-m25`、`minimax-m21` 在当前端点会返回 `400 not a multimodal model`，在用例中以 `xfail` 记录

## B1 思考模式（Thinking）

- 对应用例：
  - `tests/test_chat.py::test_create_returns_reasoning_when_thinking_enabled`
  - `tests/test_chat.py::test_stream_emits_reasoning_when_thinking_enabled`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_returns_reasoning_when_thinking_enabled or test_stream_emits_reasoning_when_thinking_enabled' -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_returns_reasoning_when_thinking_enabled or test_stream_emits_reasoning_when_thinking_enabled' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_returns_reasoning_when_thinking_enabled or test_stream_emits_reasoning_when_thinking_enabled' -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m25`、`minimax-m21`、`kimi-k25` 当前都能在非流式返回 `message.reasoning`，且最终答案包含预期的 `43`。
  - 流式链路当前也都能采集到 reasoning 增量，并在最终文本中包含 `43`。

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
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[glm-5-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[minimax-m25-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[kimi-k25-parallel_distinct_tool_calls]`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm-5 --chat-model minimax-m25 --chat-model kimi-k25
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`minimax-m25`、`kimi-k25` 可稳定返回同一条 assistant 消息中的两个不同 tool calls。
  - `qwen35` 当前会耗尽 completion 长度，只输出 reasoning，不产出结构化 `tool_calls`。
  - `minimax-m21` 当前会以文本/XML 形式写出工具调用并因长度截断，未进入结构化 `tool_calls` 通道。

## B7 工具调用-多步链式

- 对应用例：
  - `tests/test_tool_calling.py::test_multi_step_tool_chain_round_trip`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm-5 --chat-model minimax-m25 --chat-model minimax-m21 -rx
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`minimax-m21`、`minimax-m25` 可稳定完成 `fetch_seed_word -> uppercase_word -> decorate_word -> [STONE]` 的 3 步链式回填。
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
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `qwen35`、`kimi-k25` 可在 `message.content` 稳定返回合法 JSON。
  - `glm-5`、`minimax-m25`、`minimax-m21` 当前会把 JSON 放在 `message.reasoning` 且 `message.content=null`，在该用例中以 `xfail` 记录。

## B9 结构化输出

- 对应用例：
  - `tests/test_chat.py::test_structured_output_tool_returns_valid_arguments`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m25`、`minimax-m21`、`kimi-k25` 均可稳定通过。
