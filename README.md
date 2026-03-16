# maas-test

这是一个以 `httpx` 为主、以 OpenAI Python SDK 为辅的实时集成测试仓库，用来验证 vLLM 提供的 OpenAI-compatible chat completion 接口是否按预期工作。

仓库现在分成四类测试：

- 基础 chat 套件：覆盖基础 create、stream、stream + `enable_thinking=false`、StructuredOutput
- context length 套件：覆盖当前模型可发现性和上下文边界的 live 二分探测
- tool calling 套件：覆盖 httpx 工具调用主路径与代表性 stream 主路径，外加少量显式开启的 SDK probe
- SDK smoke 套件：少量官方 Python SDK 接入验证，默认跳过

## 运行前准备

仓库根目录下的 `.env` 会提供测试所需配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_CHAT_TEST_MODEL`
- `OPENAI_CHAT_TEST_MODELS`（可选，逗号分隔，用于一次运行多个模型）

安装依赖：

```bash
uv sync
```

运行默认主套件：

```bash
uv run pytest -q test_chat.py test_context_length.py test_tool_calling.py
```

运行 SDK smoke 套件：

```bash
uv run pytest -q test_chat_sdk_smoke.py --run-sdk-smoke
```

运行 tool calling probe 套件：

```bash
uv run pytest -q test_tool_calling.py --run-tool-calling-probe
```

运行默认主套件 + SDK smoke：

```bash
uv run pytest -q test_chat.py test_context_length.py test_tool_calling.py test_chat_sdk_smoke.py --run-sdk-smoke
```

运行指定模型：

```bash
uv run pytest -q test_chat.py test_context_length.py test_tool_calling.py --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25
```

也可以通过环境变量一次指定多个模型：

```bash
OPENAI_CHAT_TEST_MODELS=glm5,qwen35 uv run pytest -q test_chat.py test_context_length.py test_tool_calling.py
```

失败用例归档：

- 测试失败时，会把该用例的请求与响应详细记录到 `test_failure_artifacts/`
- 归档内容包括测试名、模型名、请求 URL、请求头、请求体、响应状态码、响应头、响应体，以及异常信息

## 基础 Chat 套件

基础 chat 套件位于 [test_chat.py](/Users/wangshilong/Downloads/maas-test/test_chat.py)。它只保留非工具调用主路径：基础 create、基础 SSE stream、`chat_template_kwargs.enable_thinking=false` 的 create / stream、以及 `StructuredOutput` 结构化输出。

### 测试行为

`test_create_returns_non_empty_assistant_message`

- 验证最基本的 `/chat/completions` JSON 请求是否成功
- 检查返回对象是否是正常的 `chat.completion`
- 检查 assistant 消息内容是否非空
- 检查 usage 统计是否返回

`test_stream_sse_emits_content_and_done`

- 验证 `stream=true` 时服务是否返回符合预期的 SSE 事件流
- 检查 `Content-Type` 是否为 `text/event-stream`
- 检查是否收到至少一个 `data:` JSON chunk
- 检查是否存在 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 是否能组成正确答案

`test_create_accepts_chat_template_kwargs_enable_thinking_false`

- 验证请求体接受 `chat_template_kwargs={"enable_thinking": false}`
- 检查带该选项的基础 `/chat/completions` 请求仍能成功返回
- 检查 assistant 文本内容仍然包含预期的 `quartz`
- 对已知稳定支持该行为的模型，额外检查 `reasoning` 会变成 `null`

`test_stream_accepts_chat_template_kwargs_enable_thinking_false`

- 验证 `stream=true` 与 `chat_template_kwargs={"enable_thinking": false}` 可以同时使用
- 检查响应仍是合法的 SSE 事件流，并带有 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 仍能组成包含 `quartz` 的最终文本
- 对已知稳定支持该行为的模型，额外检查流式增量里不会再出现 reasoning 文本

`test_structured_output_tool_returns_valid_arguments`

- 验证 `StructuredOutput` 工具风格的结构化输出路径是否可用
- 检查 `message.tool_calls` 是否存在
- 检查函数名是否为 `StructuredOutput`
- 检查工具参数 JSON 是否能被解析
- 检查返回字段是否严格符合 `word` / `length`
- 检查字段值是否与提示一致

### 模型矩阵

| 模型类 | tools 策略 | `enable_thinking=false` 行为 | StructuredOutput 策略 |
| --- | --- | --- | --- |
| `TestKimiK25ChatCompletions` | 强制命名 `tool_choice` | 请求可接受，但 `reasoning` 仍可能返回文本 | 强制命名 `StructuredOutput` 工具 |
| `TestGLM5ChatCompletions` | `tool_choice="auto"` | 请求可接受，且当前稳定返回 `reasoning=null` | `tool_choice="auto"` |
| `TestQwen35ChatCompletions` | `tool_choice="auto"` | 请求可接受，且当前稳定返回 `reasoning=null` | `tool_choice="auto"` |
| `TestMinimaxM25ChatCompletions` | `tool_choice="auto"` | 请求可接受，但 `reasoning` 仍可能返回文本 | `tool_choice="auto"` |
| `TestMinimaxM21ChatCompletions` | `tool_choice="auto"` | 请求可接受，但 `reasoning` 仍可能返回文本 | `tool_choice="auto"` |

如果命令行显式传了 `--chat-model`，基础 chat 套件只会运行对应模型类。默认模型列表位于 [chat_models.json](/Users/wangshilong/Downloads/maas-test/chat_models.json)。

基础 chat 套件的设计目标是“每个模型按已知最佳请求形状通过”，而不是强迫所有模型接受同一套最严格语义。因此，`StructuredOutput` 工具路径允许按模型类做最小差异化。

### 模型接口返回特点

| 模型 | 基础 `create` / `stream` | `enable_thinking=false` 行为 | tools 行为 | StructuredOutput 行为 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `kimi-k25` | 正常 | 请求可接受，但 `reasoning` 仍可能返回文本 | forced named `tool_choice` 可用，`message.tool_calls` 正常返回 | 强制命名 `StructuredOutput` 工具可复用同一路径 | 即使返回了 `tool_calls`，`finish_reason` 也可能是 `stop` |
| `glm5` | 正常 | 请求可接受，且当前稳定返回 `reasoning=null` | forced named `tool_choice` 会返回顶层 `error`；去掉强制 `tool_choice` 后 relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 和 relaxed tools 行为一致 |
| `qwen35` | 正常 | 请求可接受，且当前稳定返回 `reasoning=null` | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 和 `glm5` 行为接近 |
| `minimax-m25` | 正常 | 请求可接受，但 `reasoning` 仍可能返回文本 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 复用同一套工具调用最佳路径 |
| `minimax-m21` | 正常 | 请求可接受，但 `reasoning` 仍可能返回文本 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 行为基本与 `minimax-m25` 一致 |

## Context Length 套件

context length 套件位于 [test_context_length.py](/Users/wangshilong/Downloads/maas-test/test_context_length.py)。这个文件是默认主套件的一部分，不再是独立脚本。它会对当前选中的每个模型执行一次“先指数扩边、再二分收敛”的上下文边界探测。

### 测试行为

`test_context_length_finds_a_finite_boundary`

- 先请求 `/v1/models`，要求当前模型必须出现在可用模型列表里
- 用指数扩边找到“最后一个成功点”和“第一个上下文溢出点”
- 再对这两个边界做二分收敛，逼近服务实际接受与拒绝的分界位置
- chat 请求统一使用 `temperature=0` 与 `max_completion_tokens=1`
- 要求至少存在一个成功样本和一个带上下文溢出信号的失败样本
- 要求最终二分结果满足“最后成功点”的重复次数小于“第一个失败点”

### 说明

- 这个用例只要求 `/v1/models` 能列出当前模型，以及 `/chat/completions` 在超长 prompt 下返回可识别的上下文溢出错误
- 如果 `/v1/models` 没有返回当前模型，测试会直接失败
- 这个用例比基础 chat / tool calling 更重，因为每个模型都需要多次 live 请求才能完成边界收敛

## SDK Smoke 套件

SDK smoke 套件位于 [test_chat_sdk_smoke.py](/Users/wangshilong/Downloads/maas-test/test_chat_sdk_smoke.py)，默认跳过，只在显式传入 `--run-sdk-smoke` 时运行。

### 1. SDK 基础调用

`test_sdk_create_returns_non_empty_assistant_message`

验证官方 Python SDK 最基本的 `chat.completions.create()` 是否还能成功接入当前服务。

### 2. SDK 原始流式调用

`test_sdk_stream_true_yields_non_empty_text`

验证官方 Python SDK 的 `create(stream=True)` 最基本流式接入是否正常。

## Tool Calling 套件

tool calling 套件位于 [test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/test_tool_calling.py)。这个文件包含两部分：

- 默认运行的 httpx 工具调用主套件（包含代表性 stream 覆盖）
- 仅在显式传入 `--run-tool-calling-probe` 时运行的 SDK probe

### 默认 httpx 工具调用行为

`test_create_returns_tool_call`

- 验证基础 weather tool call 是否可用
- 检查 `message.tool_calls` 是否存在
- 检查函数名是否为 `collect_weather_args`
- 检查参数是否包含 `city=Tokyo` 与 `unit=celsius`

`test_stream_returns_tool_call`

- 验证非 SDK `stream=true` 的 weather tool call 是否可用
- 检查 SSE 响应头是否为 `text/event-stream`
- 检查流式聚合后的第一个工具调用函数名是否为 `collect_weather_args`
- 检查聚合后的参数 JSON 仍然包含 `city=Tokyo` 与 `unit=celsius`

`test_tool_call_round_trip_returns_final_assistant_message`

- 验证多轮 tool loop 是否可用
- 第一轮检查 assistant 是否先返回 `collect_weather_args` 工具调用
- 第二轮把 assistant 的 `tool_calls` 与对应 `tool` 结果回填到 `messages`
- 检查服务是否能继续生成最终 assistant 文本答复

`test_stream_tool_call_round_trip_returns_final_assistant_message`

- 验证非 SDK streamed tool call 后，继续回填 `tool` 结果并再次使用 `stream=true` 的两轮路径是否可用
- 第一轮从 SSE `delta.tool_calls` 增量中聚合出完整 `collect_weather_args` 调用
- 第二轮把聚合后的 `assistant.tool_calls` 与对应 `tool` 结果回填到 `messages`
- 检查最终 SSE 文本答复非空，且仍包含 `tokyo` 或 `celsius`

`test_create_returns_repeated_same_tool_calls`

- 验证 assistant 是否能在同一条消息里连续返回两个同名 `collect_weather_args` 工具调用
- 检查 `message.tool_calls` 至少包含两个条目
- 检查前两个函数名都为 `collect_weather_args`
- 检查两次参数分别匹配 `Tokyo/celsius` 与 `Shanghai/celsius`

`test_list_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `list` 工具的调用形状
- 检查函数名是否为 `list`
- 检查工具参数是否包含绝对目录路径与 `ignore` 列表

`test_read_tool_round_trip_returns_final_assistant_message`

- 模拟 OpenCode 内置 `read` 工具的两轮调用形状
- 第一轮检查 assistant 是否返回 `read` 工具调用，且参数包含 `filePath` / `offset` / `limit`
- 第二轮回填一个伪造的 `read` 工具结果
- 检查服务是否能继续生成最终 assistant 文本答复

`test_grep_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `grep` 工具的调用形状
- 检查函数名是否为 `grep`
- 检查工具参数是否包含 `pattern`、`path`、`include`

`test_bash_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `bash` 工具的调用形状
- 检查函数名是否为 `bash`
- 检查工具参数是否包含 `command`、`workdir`、`description`
- 如果模型返回了 `timeout`，检查它是整数

`test_edit_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `edit` 工具的调用形状
- 检查函数名是否为 `edit`
- 检查工具参数是否包含 `filePath`、`oldString`、`newString`
- 如果模型返回了 `replaceAll`，检查它是 `false`

`test_write_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `write` 工具的调用形状
- 检查函数名是否为 `write`
- 检查工具参数是否包含 `filePath` 与 `content`

`test_task_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `task` 工具的调用形状
- 检查函数名是否为 `task`
- 检查工具参数是否包含 `description`、`prompt`、`subagent_type`

`test_todowrite_tool_returns_valid_arguments`

- 模拟 OpenCode 内置 `todowrite` 工具的调用形状
- 检查函数名是否为 `todowrite`
- 检查工具参数是否包含合法的 `todos` 数组
- 检查每个 todo 项是否包含 `content`、`status`、`priority`

`test_create_accepts_multi_turn_history_with_assistant_and_tool_messages`

- 验证服务是否接受包含 `assistant` / `tool` 历史消息的多轮会话形状
- 复用第一轮真实生成的工具调用，构造更接近 OpenCode 的历史消息
- 在已有 `assistant + tool` 历史后追加新的 `user` 追问
- 检查服务是否仍能返回稳定的 assistant 文本

当前模型矩阵下，tool calling 套件不包含 `apply_patch`，因为 OpenCode 对这组模型更真实的默认路径仍是 `edit` / `write`。其中 `edit` 和 `task` 对 `qwen35` 不在稳定 passing path 中，因此该模型会跳过这两项，而不是强行统一请求形状。新增的“同响应内连续两次同名 weather tool call”用例对 `kimi-k25`、`qwen35` 和 `minimax-m21` 也不在稳定 passing path 中，因此这三个模型会显式跳过该测试。streamed weather tool call 及其 round-trip 默认作为主套件的一部分运行；其中 `qwen35` 的 streamed tool-call round-trip 第二轮当前只稳定返回 reasoning、不稳定产出最终文本，因此会显式跳过该用例。如果某个模型经过 live 验证后不稳定，应优先在模型类里显式关闭对应 stream 用例，而不是让默认主套件随机失败。

### SDK Probe

下面这些 probe 用例默认跳过，只在显式传入 `--run-tool-calling-probe` 时运行。

### 1. 单工具非流式调用

`test_sdk_single_tool_call_returns_valid_json_arguments`

验证官方 Python SDK 的非流式 tool call 是否返回合法 JSON 参数。

### 2. 单工具流式调用

`test_sdk_stream_tool_call_emits_valid_json_arguments`

验证官方 Python SDK 的流式 tool call chunk 聚合后是否仍能组成合法 JSON 参数。
其中 `qwen35` 的 streamed tool-call 不在稳定 passing path 中，因此会跳过。

### 3. 多工具请求

`test_sdk_multi_tool_request_returns_valid_tool_calls`

验证在同一请求中提供多个工具定义时，模型返回的 tool call 名称和参数 JSON 是否保持有效。

### 4. 大参数负载

`test_sdk_large_tool_arguments_remain_valid_json`

验证较大的 tool call arguments 在 SDK 路径下仍然保持 JSON 可解析。

如果某个用例失败，可以直接打开 [test_failure_artifacts](/Users/wangshilong/Downloads/maas-test/test_failure_artifacts) 里的最新归档文件复盘请求和响应细节。
