# Test Run Commands

`main.py` 会把这个文件当作机器可读的命令源来解析。维护约定如下：

- 每个 `## <CASE_ID>` 小节都应保留一个明确标注的默认命令。
- 标签里包含 `默认` 的命令会被 `main.py` 优先选择。
- 如果同一小节出现多个 `默认...命令`，`main.py` 当前会只取第一个，并在 `run_manifest.json` 里写入告警。
- `单模型复测示例` 默认只用于人工复查；只有当该小节没有其他可运行命令时，`main.py` 才会退回使用它。

## H1 Chat Completions 兼容

- 对应用例：
  - `tests/test_api_compatibility.py::test_chat_completions_returns_openai_shape_and_usage`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_chat_completions_returns_openai_shape_and_usage --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m2.5`、`minimax-m21` 当前可返回 OpenAI-compatible 的 `chat.completion` 结构
  - `kimi-k25` 在当前环境可能直接命中上游 DNS/路由型 `500 do_request_failed`，导致 H1 在响应形状断言前失败

## H2 Raw Completions 兼容

- 对应用例：
  - `tests/test_api_compatibility.py::test_raw_completions_returns_openai_shape_and_usage`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_raw_completions_returns_openai_shape_and_usage --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m2.5`、`minimax-m21` 当前可返回 `text_completion` 结构，并带 `usage`
  - `kimi-k25` 在当前环境可能直接命中上游 DNS/路由型 `500 do_request_failed`，导致 H2 在响应形状断言前失败

## H3 /v1/models 模型列表

- 对应用例：
  - `tests/test_api_compatibility.py::test_models_endpoint_lists_selected_model`
- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_api_compatibility.py -k test_models_endpoint_lists_selected_model --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
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
uv run pytest -q tests/test_api_compatibility.py -k 'test_chat_completions_returns_openai_shape_and_usage or test_raw_completions_returns_openai_shape_and_usage' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m2.5`、`minimax-m21` 当前在 `/v1/chat/completions` 与 `/v1/completions` 两条链路都满足 `prompt_tokens + completion_tokens == total_tokens`
  - `kimi-k25` 若继续返回上游 DNS/路由型 `500`，则该轮无法完成 H4 的 usage 算术校验

## A1 单轮对话

- 对应用例：
  - `tests/test_chat.py::test_create_returns_non_empty_assistant_message`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_returns_non_empty_assistant_message -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_returns_non_empty_assistant_message --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_returns_non_empty_assistant_message -rx
```

## A2 多轮对话

- 对应用例：
  - `tests/test_chat.py::test_create_preserves_multi_turn_context`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_preserves_multi_turn_context -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_preserves_multi_turn_context --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_preserves_multi_turn_context -rx
```

## A3 System Prompt

- 对应用例：
  - `tests/test_chat.py::test_create_respects_system_prompt_priority`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_respects_system_prompt_priority -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_respects_system_prompt_priority --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_respects_system_prompt_priority -rx
```

## A4 流式输出

- 对应用例：
  - `tests/test_chat.py::test_stream_sse_emits_content_and_done`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_stream_sse_emits_content_and_done -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_stream_sse_emits_content_and_done --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_stream_sse_emits_content_and_done -rx
```

## A5 非流式输出

- 对应用例：
  - `tests/test_chat.py::test_create_returns_non_empty_assistant_message`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_returns_non_empty_assistant_message -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_returns_non_empty_assistant_message --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_returns_non_empty_assistant_message -rx
```

## A8 Max Tokens 限制

- 对应用例：
  - `tests/test_chat.py::test_create_respects_max_completion_tokens_limit`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_respects_max_completion_tokens_limit -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_respects_max_completion_tokens_limit --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_respects_max_completion_tokens_limit -rx
```

## A11 多语言能力

- 对应用例：
  - `tests/test_chat.py::test_create_supports_multilingual_output`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_supports_multilingual_output -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_supports_multilingual_output --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_supports_multilingual_output -rx
```

## A12 特殊 Token 处理

- 对应用例：
  - `tests/test_chat.py::test_create_preserves_special_tokens_in_text`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_preserves_special_tokens_in_text -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_preserves_special_tokens_in_text --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_preserves_special_tokens_in_text -rx
```

## C1 单图理解

- 对应用例：
  - `tests/test_chat.py::test_create_understands_single_image_dominant_color`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_understands_single_image_dominant_color -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_create_understands_single_image_dominant_color --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k test_create_understands_single_image_dominant_color -rx
```

- 当前已知现象：
  - `qwen35`、`kimi-k25` 可稳定识别内置纯红色图片并返回 `red`
  - `glm-5`、`minimax-m2.5`、`minimax-m21` 在当前端点会返回 `400 not a multimodal model`，在用例中以 `xfail` 记录

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
uv run pytest -q tests/test_chat.py -k 'test_create_returns_reasoning_when_thinking_enabled or test_stream_emits_reasoning_when_thinking_enabled' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_returns_reasoning_when_thinking_enabled or test_stream_emits_reasoning_when_thinking_enabled' -rx
```

- 当前已知现象：
  - 非流式链路会返回 `message.reasoning` / `message.reasoning_content`，或直接把解释性文本写进 `message.content`；同时要求最终答案包含预期的 `43`。
  - 流式链路优先采集 `delta.reasoning` / `delta.reasoning_content`；若未单独输出 reasoning 增量，则以最终拼接文本中出现非纯数字的解释性内容作为替代信号，并同时要求包含 `43`。

## B2 非思考模式（Instant）

- B2-A 请求可接受性
- 对应用例：
  - `tests/test_chat.py::test_create_accepts_chat_template_kwargs_enable_thinking_false`
  - `tests/test_chat.py::test_stream_accepts_chat_template_kwargs_enable_thinking_false`
- 默认模型矩阵复测命令（同时覆盖 B2-A + B2-B）：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_accepts_chat_template_kwargs_enable_thinking_false or test_stream_accepts_chat_template_kwargs_enable_thinking_false or test_create_suppresses_reasoning_when_thinking_disabled or test_stream_suppresses_reasoning_when_thinking_disabled' -rxX
```

- B2-B 严格 suppress hidden thinking
- 对应用例：
  - `tests/test_chat.py::test_create_suppresses_reasoning_when_thinking_disabled`
  - `tests/test_chat.py::test_stream_suppresses_reasoning_when_thinking_disabled`
- B2-B 单独复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_suppresses_reasoning_when_thinking_disabled or test_stream_suppresses_reasoning_when_thinking_disabled' -rxX
```

- B2-A 单独复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_accepts_chat_template_kwargs_enable_thinking_false or test_stream_accepts_chat_template_kwargs_enable_thinking_false'
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_accepts_chat_template_kwargs_enable_thinking_false or test_stream_accepts_chat_template_kwargs_enable_thinking_false'
```

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_suppresses_reasoning_when_thinking_disabled or test_stream_suppresses_reasoning_when_thinking_disabled' -rxX
```

- 当前已知现象：
  - B2-B 现在按三通道 stream 聚合执行：`delta.content` 必须非空，`delta.reasoning` 与 `delta.reasoning_content` 都必须为空。
  - 不再依赖最终答案文本是否等于 `quartz`；流式 suppress 只按通道占用关系判定。

## B3 思考模式切换

- 对应用例：
  - `tests/test_chat.py::test_create_switches_thinking_to_instant_within_same_conversation`
  - `tests/test_chat.py::test_create_switches_instant_to_thinking_within_same_conversation`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_switches_thinking_to_instant_within_same_conversation or test_create_switches_instant_to_thinking_within_same_conversation' -rxX
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k 'test_create_switches_thinking_to_instant_within_same_conversation or test_create_switches_instant_to_thinking_within_same_conversation' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rxX
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_chat.py --chat-model kimi-k25 -k 'test_create_switches_thinking_to_instant_within_same_conversation or test_create_switches_instant_to_thinking_within_same_conversation' -rxX
```

- 当前已知现象：
  - 首次接入：用于验证同一 messages history 下 `chat_template_kwargs.enable_thinking` 从 `true↔false` 切换的请求可用性。
  - `enable_thinking=false` 下的严格 suppress hidden thinking 口径与 B2 保持一致：若仍返回 `reasoning` / `reasoning_content`，或把 explanation 混进最终 `content`，则以 `xfail` 记录。

## B4 工具调用-单工具

- 对应用例：
  - `tests/fixtures/k2/tool_calling_subset.jsonl` 中的 `single_tool_nonstream`
  - `tests/fixtures/k2/tool_calling_subset.jsonl` 中的 `single_tool_stream`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k 'single_tool_nonstream or single_tool_stream' -rx
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k 'single_tool_nonstream or single_tool_stream' --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 单模型复测示例：

```bash
uv run pytest -q tests/test_tool_calling.py --chat-model kimi-k25 -k 'single_tool_nonstream or single_tool_stream' -rx
```

- 当前已知现象：
  - `kimi-k25` 当前可稳定通过。
  - `glm-5`、`qwen35`、`minimax-m21`、`minimax-m2.5` 在当前后端上可能复现 `500` / `upstream_error` 等失败（以实时结果为准）。

## B6 工具调用-并行调用

- 对应用例：
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[glm-5-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[minimax-m2.5-parallel_distinct_tool_calls]`
  - `tests/test_tool_calling.py::test_dataset_driven_tool_calling_case[kimi-k25-parallel_distinct_tool_calls]`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm-5 --chat-model minimax-m2.5 --chat-model kimi-k25
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k parallel_distinct_tool_calls --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`minimax-m2.5`、`kimi-k25` 可稳定返回同一条 assistant 消息中的两个不同 tool calls。
  - `qwen35` 当前会耗尽 completion 长度，只输出 reasoning，不产出结构化 `tool_calls`。
  - `minimax-m21` 当前会以文本/XML 形式写出工具调用并因长度截断，未进入结构化 `tool_calls` 通道。

## B7 工具调用-多步链式

- 对应用例：
  - `tests/test_tool_calling.py::test_multi_step_tool_chain_round_trip`
- 默认稳定路径复测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm-5 --chat-model minimax-m2.5 --chat-model minimax-m21 -rx
```

- 全模型探测命令：

```bash
uv run pytest -q tests/test_tool_calling.py -k test_multi_step_tool_chain_round_trip --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`minimax-m21`、`minimax-m2.5` 可稳定完成 `fetch_seed_word -> uppercase_word -> decorate_word -> [STONE]` 的 3 步链式回填。
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
uv run pytest -q tests/test_chat.py -k test_json_mode_returns_valid_json_object --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `qwen35`、`kimi-k25` 可在 `message.content` 稳定返回合法 JSON。
  - `glm-5`、`minimax-m2.5`、`minimax-m21` 当前会把 JSON 放在 `message.reasoning` / `message.reasoning_content` 且 `message.content=null`，在该用例中以 `xfail` 记录。

## B9 结构化输出

- 对应用例：
  - `tests/test_chat.py::test_structured_output_tool_returns_valid_arguments`
- 默认模型矩阵复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments
```

- 全模型显式复测命令：

```bash
uv run pytest -q tests/test_chat.py -k test_structured_output_tool_returns_valid_arguments --chat-model glm-5 --chat-model qwen35 --chat-model minimax-m2.5 --chat-model minimax-m21 --chat-model kimi-k25 -rx
```

- 当前已知现象：
  - `glm-5`、`qwen35`、`minimax-m2.5`、`minimax-m21`、`kimi-k25` 均可稳定通过。

## L1 超长上下文 (脚本验证)

- 对应脚本：
  - `tests/test_long_context.py`
- 默认命令 (内网地址常见需绕过代理)：

```bash
NO_PROXY=172.16.84.27 uv run tests/test_long_context.py --no-proxy --url http://172.16.84.27:8080/v1 --model minimax-m2.5 --tokenizer MiniMaxAI/MiniMax-M2.5
```

- 流式观察 `reasoning_content` / `content` 增量：

```bash
NO_PROXY=172.16.84.27 uv run tests/test_long_context.py --no-proxy --stream --url http://172.16.84.27:8080/v1 --model minimax-m2.5 --tokenizer MiniMaxAI/MiniMax-M2.5 --target-tokens 128000
```
