# maas-test

这是一个以 `httpx` 为主、以 OpenAI Python SDK 为辅的实时集成测试仓库，用来验证 vLLM 提供的 OpenAI-compatible chat completion 接口是否按预期工作。

仓库现在分成三层测试：

- 默认主套件：按模型最优路径组织的类测试，目标是每个模型按自己的最佳请求形状跑通
- strict 对照套件：严格按 OpenAI-compatible 语义验证，默认跳过，只在显式开启时运行
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
uv run pytest -q test_chat.py
```

运行 strict 对照套件：

```bash
uv run pytest -q test_chat_strict.py --run-strict-compat
```

运行 SDK smoke 套件：

```bash
uv run pytest -q test_chat_sdk_smoke.py --run-sdk-smoke
```

运行包含 strict + SDK smoke 的完整测试：

```bash
uv run pytest -q --run-strict-compat --run-sdk-smoke
```

运行指定模型：

```bash
uv run pytest -q test_chat.py --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25
```

也可以通过环境变量一次指定多个模型：

```bash
OPENAI_CHAT_TEST_MODELS=glm5,qwen35 uv run pytest -q test_chat.py
```

失败用例归档：

- 测试失败时，会把该用例的请求与响应详细记录到 `test_failure_artifacts/`
- 归档内容包括测试名、模型名、请求 URL、请求头、请求体、响应状态码、响应头、响应体，以及异常信息

## 默认主套件

默认主套件位于 [test_chat.py](/Users/wangshilong/Downloads/maas-test/test_chat.py)。它采用“基类定义行为 + 模型子类覆盖差异”的方式组织，所以 `test_chat.py` 中的测试方法只定义一次，但会被不同模型类复用。

### 测试行为

`test_create_returns_non_empty_assistant_message`

- 验证最基本的 `/chat/completions` JSON 请求是否成功
- 检查返回对象是否是正常的 `chat.completion`
- 检查 assistant 消息内容是否非空
- 检查 usage 统计是否返回

`test_create_returns_tool_call`

- 验证模型的工具调用路径是否可用
- 默认主套件会按模型类选择合适的 `tool_choice` 形状；对 `glm5`、`qwen35`、`minimax-*` 会显式使用 `tool_choice="auto"`
- 检查 `message.tool_calls` 是否存在
- 检查函数名是否为 `collect_weather_args`
- 检查函数参数是否包含 `city=Tokyo` 与 `unit=celsius`

`test_stream_sse_emits_content_and_done`

- 验证 `stream=true` 时服务是否返回符合预期的 SSE 事件流
- 检查 `Content-Type` 是否为 `text/event-stream`
- 检查是否收到至少一个 `data:` JSON chunk
- 检查是否存在 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 是否能组成正确答案

`test_response_format_returns_structured_output`

- 验证结构化输出路径是否可用
- 检查结构化 JSON 是否能被解析
- 检查返回字段是否严格符合 `word` / `length`
- 检查字段值是否与提示一致

### 模型矩阵

| 模型类 | tools 策略 | `response_format` 策略 | 结构化结果通道 |
| --- | --- | --- | --- |
| `TestKimiK25ChatCompletions` | 强制命名 `tool_choice` | 原始请求 | `message.content` |
| `TestGLM5ChatCompletions` | `tool_choice="auto"` | `chat_template_kwargs.enable_thinking=false` | `message.content` |
| `TestQwen35ChatCompletions` | `tool_choice="auto"` | `chat_template_kwargs.enable_thinking=false` | `message.content` |
| `TestMinimaxM25ChatCompletions` | `tool_choice="auto"` | 原始请求 | `message.reasoning` |
| `TestMinimaxM21ChatCompletions` | `tool_choice="auto"` | 原始请求 | `message.reasoning` |

如果命令行显式传了 `--chat-model`，默认主套件只会运行对应模型类。默认模型列表位于 [chat_models.json](/Users/wangshilong/Downloads/maas-test/chat_models.json)。

默认主套件的设计目标是“每个模型按已知最佳请求形状通过”，而不是强迫所有模型接受同一套最严格语义。因此，tools 路径和结构化输出通道允许按模型类做最小差异化。

### 模型接口返回特点

| 模型 | 基础 `create` / `stream` | tools 行为 | `response_format` 行为 | 备注 |
| --- | --- | --- | --- | --- |
| `kimi-k25` | 正常 | forced named `tool_choice` 可用，`message.tool_calls` 正常返回 | JSON 在 `message.content` | 即使返回了 `tool_calls`，`finish_reason` 也可能是 `stop` |
| `glm5` | 正常 | forced named `tool_choice` 会返回顶层 `error`；去掉强制 `tool_choice` 后 relaxed tools 可用 | 默认 JSON 在 `message.reasoning`；加 `chat_template_kwargs.enable_thinking=false` 后可回到 `message.content` | 更适合“relaxed tools + 关闭 thinking” |
| `qwen35` | 正常 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 默认 JSON 在 `message.reasoning`；加 `chat_template_kwargs.enable_thinking=false` 后可回到 `message.content` | 和 `glm5` 行为接近 |
| `minimax-m25` | 正常 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | JSON 在 `message.reasoning`；`chat_template_kwargs.enable_thinking=false` 无法挪回 `message.content` | 结构化结果只能按 `reasoning` 通道验 |
| `minimax-m21` | 正常 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | JSON 在 `message.reasoning`；`chat_template_kwargs.enable_thinking=false` 无法挪回 `message.content` | 行为基本与 `minimax-m25` 一致 |

## Strict 对照套件

strict 对照套件位于 [test_chat_strict.py](/Users/wangshilong/Downloads/maas-test/test_chat_strict.py)，默认跳过，只在显式传入 `--run-strict-compat` 时运行。它的目标不是“让所有模型全绿”，而是持续暴露严格 OpenAI-compatible 语义上的偏差，例如：

- forced named `tool_choice` 是否真的可用
- `response_format` 结果是否真的出现在 `message.content`
- 流式返回是否保持标准 SSE 行为

换句话说：

- 默认主套件偏向“按模型最佳实践接入”
- strict 对照套件偏向“按统一 OpenAI-compatible 语义审计”

## SDK Smoke 套件

SDK smoke 套件位于 [test_chat_sdk_smoke.py](/Users/wangshilong/Downloads/maas-test/test_chat_sdk_smoke.py)，默认跳过，只在显式传入 `--run-sdk-smoke` 时运行。

### 1. SDK 基础调用

`test_sdk_create_returns_non_empty_assistant_message`

验证官方 Python SDK 最基本的 `chat.completions.create()` 是否还能成功接入当前服务。

### 2. SDK 原始流式调用

`test_sdk_stream_true_yields_non_empty_text`

验证官方 Python SDK 的 `create(stream=True)` 最基本流式接入是否正常。

如果某个用例失败，可以直接打开 [test_failure_artifacts](/Users/wangshilong/Downloads/maas-test/test_failure_artifacts) 里的最新归档文件复盘请求和响应细节。
