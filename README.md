# maas-test

这是一个结合 `httpx` 与 OpenAI Python SDK 的实时集成测试仓库，用来验证 vLLM 提供的 OpenAI-compatible chat completion 接口是否按预期工作。

仓库现在分成四类默认测试，外加一条显式开启的 K2 verifier 通道：

- 基础 chat 套件：覆盖基础 create、System Prompt、多轮对话、stream、max token 限制、多语言、特殊 token、thinking mode、`enable_thinking=false` 的请求可接受性与严格 suppress reasoning 校验、StructuredOutput
- context length 套件：覆盖当前模型可发现性和上下文边界的 live 二分探测
- tool calling 套件：覆盖基于 K2 sample 子集的数据集驱动 tool-calling 回放，包括单工具、同消息重复 tool call、同消息并行多工具等路径，默认走 SDK + `httpx` transport
- SDK smoke 套件：少量官方 Python SDK 接入验证，默认纳入主执行路径
- K2 verifier：面向外部 JSONL 数据集的大规模 tool-calling 精度验证，默认不进入主套件

## 运行前准备

仓库根目录下的 `.env` 会提供测试所需配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_CHAT_TEST_MODEL`
- `OPENAI_CHAT_TEST_MODELS`（可选，逗号分隔，用于一次运行多个模型）

也可以直接通过 pytest 命令行参数覆盖连接信息：

- `--OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- `--OPENAI_API_KEY=xxx`

其中 `--OPENAI_API_KEY` 如果不传，会自动设为空字符串；`--OPENAI_BASE_URL` 如果不传，则继续使用 `.env` 中的值，若 `.env` 也没有配置，则回退到 `https://api.openai.com/v1`。

安装依赖：

```bash
uv sync
```

如果只想查看 K2 verifier 和测试目录，相关路径如下：

- [k2_verifier/cli.py](/Users/wangshilong/Downloads/maas-test/k2_verifier/cli.py)：CLI 入口
- [k2_verifier/core.py](/Users/wangshilong/Downloads/maas-test/k2_verifier/core.py)：请求预处理、stream 聚合、schema 校验、结果汇总
- [tests/fixtures/k2/tool_calling_subset.jsonl](/Users/wangshilong/Downloads/maas-test/tests/fixtures/k2/tool_calling_subset.jsonl)：默认 `pytest` 工具调用子集
- [datasets/k2/smoke.jsonl](/Users/wangshilong/Downloads/maas-test/datasets/k2/smoke.jsonl)：手工 verifier smoke 数据集
- [datasets/k2/vendor_samples.jsonl](/Users/wangshilong/Downloads/maas-test/datasets/k2/vendor_samples.jsonl)：手工 verifier 大样本数据集
- [third_party/k2_vendor_verifier](/Users/wangshilong/Downloads/maas-test/third_party/k2_vendor_verifier)：上游 `K2-Vendor-Verifier` 参考快照

运行默认主套件：

```bash
uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py
```

显式指定连接地址：

```bash
uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py --OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

运行 SDK smoke 套件：

```bash
uv run pytest -q tests/test_chat_sdk_smoke.py
```

`--run-tool-calling-probe` 仍保留为兼容参数，但当前已经没有单独的 probe 套件；默认 `tests/test_tool_calling.py` 已经统一切到数据集驱动的 SDK 主路径。

运行指定模型：

```bash
uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py --chat-model glm5 --chat-model qwen35 --chat-model minimax-m25 --chat-model minimax-m21 --chat-model kimi-k25
```

也可以通过环境变量一次指定多个模型：

```bash
OPENAI_CHAT_TEST_MODELS=glm5,qwen35 uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py
```

输出可读 CSV 报告：

```bash
uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py --csv-report-dir=reports
```

运行指定模型并输出 CSV：

```bash
uv run pytest -q tests/test_chat.py tests/test_context_length.py tests/test_tool_calling.py tests/test_chat_sdk_smoke.py --chat-model glm5 --chat-model qwen35 --csv-report-dir=reports
```

失败用例归档：

- 测试失败时，会把该用例的请求与响应详细记录到 `test_failure_artifacts/`
- 归档内容包括测试名、模型名、请求 URL、请求头、请求体、响应状态码、响应头、响应体，以及异常信息

CSV 报告输出：

- 传入 `--csv-report-dir=PATH` 后，会在该目录生成 `results.csv`、`summary.csv` 与 `stats.csv`
- `results.csv` 按“每个实际执行的 pytest item 一行”输出，适合按测试名、模型名、结果筛选
- `summary.csv` 按“每个 case 一行”输出，列头格式固定为 `测试类型, case 名, 测试内容, <model>测试结果, ...`，适合直接做多模型对比
- `summary.csv` 只保留本次显式选中模型的结果列，不会额外生成 `unknown测试结果` 这类内部占位列
- `stats.csv` 保留按 `suite + model` 聚合的通过率统计，同时包含 `ALL + model` 与 `ALL + ALL` 总览行，适合快速看整轮健康度
- CSV 只记录实际执行项；因模型过滤而 `deselected` 的项不会单独写入 CSV
- 失败行只保留短摘要与 `failure_artifact` 路径，完整请求/响应细节继续看 `test_failure_artifacts/`

`results.csv` 主要列：

- `suite`: 来源测试文件，例如 `test_chat.py`
- `test_name`: 原始 pytest 测试函数名
- `description`: 可直接阅读的测试行为说明
- `nodeid`: 完整 pytest 节点 ID，便于精确定位
- `model`: 当前执行项对应的模型名；不依赖模型时为 `unknown`
- `outcome`: `passed` / `failed` / `skipped`
- `duration_seconds`: 该执行项耗时
- `base_url`: 当前请求使用的 OpenAI-compatible endpoint
- `selected_models`: 本次运行实际选中的模型列表，逗号分隔
- `failure_summary`: 单行失败或跳过摘要
- `failure_artifact`: 失败归档 JSON 路径；成功时为空

`summary.csv` 主要列：

- `测试类型`: 默认等于来源测试文件，例如 `test_chat.py`
- `case 名`: pytest 测试函数名
- `测试内容`: 可直接阅读的测试行为说明
- `<model>测试结果`: 每个模型各占一列，值为 `passed` / `failed` / `skipped` / `not_run`

`stats.csv` 主要列：

- `suite` 与 `model`: 汇总维度
- `passed` / `failed` / `skipped` / `total`: 对应维度下的统计计数
- `pass_rate`: 通过率，按 `passed / total` 计算

## K2 Verifier

K2 verifier 是一条独立的补充评测通道，用来对外部 JSONL 请求集做大规模 tool-calling 验证。它不会进入默认 `pytest` 主套件，也不会写入 `test_failure_artifacts/`。

### 适用场景

- 验证某个 OpenAI-compatible 部署在大量 tool-calling 样本上的 `finish_reason` 与 schema 准确率
- 对比不同 provider / guided encoding / `chat_template_kwargs` 参数组合
- 复现上游 [K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) 的核心执行形状，同时继续复用本仓库的 `.env`

### 运行方式

最小运行命令：

```bash
uv run python -m k2_verifier.cli path/to/dataset.jsonl --model kimi-k25
```

显式指定 endpoint：

```bash
uv run python -m k2_verifier.cli path/to/dataset.jsonl \
  --model kimi-k25 \
  --base-url http://127.0.0.1:8000/v1
```

传入额外请求体参数：

```bash
uv run python -m k2_verifier.cli path/to/dataset.jsonl \
  --model kimi-k25 \
  --extra-body '{"chat_template_kwargs":{"thinking":false},"temperature":0.6}'
```

指定输出文件：

```bash
uv run python -m k2_verifier.cli path/to/dataset.jsonl \
  --model kimi-k25 \
  --output reports/k2vv/manual/results.jsonl \
  --summary reports/k2vv/manual/summary.json
```

如果不显式传 `--base-url`，会继续复用 `.env` 或环境变量里的 `OPENAI_BASE_URL`。如果不传 `--api-key`，会继续复用 `.env` 或环境变量里的 `OPENAI_API_KEY`。

官方 sample dataset 下载地址：

```bash
mkdir -p /tmp/k2vv-sample
cd /tmp/k2vv-sample
curl -L --fail -o tool-calls.tar.gz https://statics.moonshot.cn/k2vv/tool-calls.tar.gz
tar -xzf tool-calls.tar.gz
```

解压后可直接使用的样例文件路径是 `tool-calls/samples.jsonl`，例如：

```bash
uv run python -m k2_verifier.cli /tmp/k2vv-sample/tool-calls/samples.jsonl --model kimi-k25
```

基于当前仓库配置、可直接复制执行的完整命令：

```bash
mkdir -p /tmp/k2vv-sample && \
cd /tmp/k2vv-sample && \
curl -L --fail -o tool-calls.tar.gz https://statics.moonshot.cn/k2vv/tool-calls.tar.gz && \
tar -xzf tool-calls.tar.gz && \
cd /Users/wangshilong/Downloads/maas-test && \
uv run python -m k2_verifier.cli /tmp/k2vv-sample/tool-calls/samples.jsonl \
  --model kimi-k25 \
  --base-url https://codingplan-staging.alayanew.com:26443/v1 \
  --concurrency 5 \
  --output reports/k2vv/manual/results.jsonl \
  --summary reports/k2vv/manual/summary.json
```

这条命令默认继续复用仓库根目录 `.env` 里的 `OPENAI_API_KEY`。

### 输入与兼容性

- 输入文件必须是 JSONL；每一行都应是完整的 OpenAI-compatible 请求体
- 默认走 `/v1/chat/completions`
- 传入 `--use_raw_completions` 时会改走 `/v1/completions`，并使用 tokenizer 把 `messages` 转成 prompt；该模式需要本地安装 `transformers`
- 默认会把历史 `assistant.tool_calls[].id` 规范化为 `functions.<name>:<idx>`，并同步修正对应的 `tool.tool_call_id`
- 传入 `--disable-tool-call-id-normalization` 可关闭这一步兼容处理
- `--extra-body` 会透传给 OpenAI SDK 的 `extra_body`，适合传 provider 路由、guided encoding、`chat_template_kwargs` 等后端特定参数

### 输出内容

未显式指定 `--output` / `--summary` 时，结果默认写到 `reports/k2vv/<timestamp>/`：

- `results.jsonl`：逐请求结果，包含准备后的请求体、响应体、`finish_reason`、`tool_calls_valid`、耗时和稳定 hash
- `summary.json`：聚合统计，包含 `success_count`、`failure_count`、`finish_tool_calls`、`successful_tool_call_count`、`schema_validation_error_count` 与 usage 汇总；每处理完一条请求都会覆盖更新一次，因此中断运行时也能看到部分统计，完成前 `eval_finished_at` 会保持 `null`

## 基础 Chat 套件

基础 chat 套件位于 [tests/test_chat.py](/Users/wangshilong/Downloads/maas-test/tests/test_chat.py)。它只保留非工具调用主路径：基础 create、System Prompt 遵循、多轮对话上下文保持、基础 SSE stream、图片 content parts 的 create / stream、`max_completion_tokens` 限制、多语言输出、特殊 token 保留、thinking mode 的 create / stream、`chat_template_kwargs.enable_thinking=false` 的 create / stream 请求可接受性、该选项下的严格 suppress reasoning 校验，以及 `StructuredOutput` 结构化输出。

### 测试行为

`test_create_returns_non_empty_assistant_message`

- 验证最基本的 `/chat/completions` JSON 请求是否成功
- 检查返回对象是否是正常的 `chat.completion`
- 检查 assistant 消息内容是否非空
- 检查 usage 统计是否返回

`test_create_respects_system_prompt_priority`

- 验证 system prompt 与 user 指令冲突时，模型仍优先遵循 system prompt
- 构造一个要求返回 `system-wins` 的 system 消息，以及一个要求返回 `user-wins` 的冲突 user 消息
- 检查最终回答是否仍然是 `system-wins`

`test_create_preserves_multi_turn_context`

- 验证普通多轮对话历史消息可被接受
- 构造包含 `user -> assistant -> user` 历史的单次 `/chat/completions` 请求
- 检查最后一轮回答是否能正确回忆前文给出的 `bamboo-7`
- 检查返回对象与 usage 统计仍然正常

`test_stream_sse_emits_content_and_done`

- 验证 `stream=true` 时服务是否返回符合预期的 SSE 事件流
- 检查 `Content-Type` 是否为 `text/event-stream`
- 检查是否收到至少一个 `data:` JSON chunk
- 检查是否存在 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 是否能组成正确答案

`test_create_accepts_image_content_parts`

- 验证 `/chat/completions` 接受 OpenAI-compatible 的 content parts 输入（`text` + `image_url`）
- 图片使用内置 `data:image/png;base64,...`，避免依赖外部网络
- 检查返回对象是否是正常的 `chat.completion`
- 检查 assistant 消息内容是否非空
- 检查 usage 统计是否返回

`test_create_respects_max_completion_tokens_limit`

- 验证 `max_completion_tokens` 可以限制输出长度
- 发送一个明显会超出限制的计数请求，并把 `max_completion_tokens` 压到较小值
- 检查响应仍然成功返回，且 `usage.completion_tokens` 不超过限制

`test_create_supports_multilingual_output`

- 验证模型可按要求输出中、英、日、韩、法混合文本
- 使用中文指令要求原样输出固定的多语言字符串
- 检查返回文本与预期多语言内容一致

`test_create_preserves_special_tokens_in_text`

- 验证文本中的 emoji、HTML 标签、代码片段和数学符号不会在简单回显场景下被破坏
- 检查返回内容中仍包含 `😀`、`<div>ok</div>`、`` `x=1` `` 和 `∑`

`test_stream_accepts_image_content_parts`

- 验证 `stream=true` 时也能接受 content parts 输入（`text` + `image_url`）
- 检查 `Content-Type` 是否为 `text/event-stream`
- 检查是否收到至少一个 `data:` JSON chunk
- 检查是否存在 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 是否非空，并包含提示中的 `quartz`

`test_create_accepts_chat_template_kwargs_enable_thinking_false`

- 验证请求体接受 `chat_template_kwargs={"enable_thinking": false}`
- 检查带该选项的基础 `/chat/completions` 请求仍能成功返回
- 检查 assistant 文本内容仍然包含预期的 `quartz`

`test_create_suppresses_reasoning_when_thinking_disabled`

- 严格验证 `chat_template_kwargs={"enable_thinking": false}` 的非流式返回不再携带冗余 `reasoning`
- 当 `message.reasoning` 缺失、为 `null`、或仅为空白字符串时视为通过
- 对当前已知仍会返回 `reasoning` 的模型，记录为 `xfail`，便于和“请求可接受”分开观察

`test_create_returns_reasoning_when_thinking_enabled`

- 验证请求体接受 `chat_template_kwargs={"enable_thinking": true}`
- 检查非流式返回仍能给出最终答案
- 检查 `message.reasoning` 存在且为非空字符串
- 通过固定算术题把最终答案约束到包含 `43`

`test_stream_emits_reasoning_when_thinking_enabled`

- 验证 `stream=true` 与 `chat_template_kwargs={"enable_thinking": true}` 可以同时使用
- 检查响应仍是合法的 SSE 事件流，并带有 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 最终答案包含 `43`
- 检查流式增量里能采集到非空的 reasoning 片段

`test_stream_accepts_chat_template_kwargs_enable_thinking_false`

- 验证 `stream=true` 与 `chat_template_kwargs={"enable_thinking": false}` 可以同时使用
- 检查响应仍是合法的 SSE 事件流，并带有 `[DONE]` 终止事件
- 检查拼接后的 `delta.content` 仍能组成包含 `quartz` 的最终文本

`test_stream_suppresses_reasoning_when_thinking_disabled`

- 严格验证 `stream=true` 且 `chat_template_kwargs={"enable_thinking": false}` 时，流式增量里不会再出现冗余 `reasoning`
- SSE 聚合后的 `stream_result.reasoning` 为 `null` 视为通过
- 对当前已知仍会返回 reasoning 片段的模型，记录为 `xfail`，便于单独追踪 suppress reasoning 行为

`test_structured_output_tool_returns_valid_arguments`

- 验证 `StructuredOutput` 工具风格的结构化输出路径是否可用
- 检查 `message.tool_calls` 是否存在
- 检查函数名是否为 `StructuredOutput`
- 检查工具参数 JSON 是否能被解析
- 检查返回字段是否严格符合 `word` / `length`
- 检查字段值是否与提示一致

### 模型矩阵

| 模型类 | 基础 create/stream 请求 | tools 策略 | `enable_thinking=false` 行为 | StructuredOutput 策略 |
| --- | --- | --- | --- | --- |
| `TestKimiK25ChatCompletions` | 默认请求 | 强制命名 `tool_choice` | 请求可接受；严格 suppress reasoning 测试下仍可能返回 `reasoning` | 强制命名 `StructuredOutput` 工具 |
| `TestGLM5ChatCompletions` | 默认请求 | `tool_choice="auto"` | 请求可接受，且严格 suppress reasoning 测试当前稳定返回 `reasoning=null` | `tool_choice="auto"` |
| `TestQwen35ChatCompletions` | 基础文本请求默认附带 `chat_template_kwargs.enable_thinking=false` | `tool_choice="auto"` | 请求可接受，且严格 suppress reasoning 测试当前稳定返回 `reasoning=null` | `tool_choice="auto"` |
| `TestMinimaxM25ChatCompletions` | 默认请求 | `tool_choice="auto"` | 请求可接受；严格 suppress reasoning 测试下仍可能返回 `reasoning` | `tool_choice="auto"` |
| `TestMinimaxM21ChatCompletions` | 默认请求 | `tool_choice="auto"` | 请求可接受；严格 suppress reasoning 测试下仍可能返回 `reasoning` | `tool_choice="auto"` |

如果命令行显式传了 `--chat-model`，基础 chat 套件只会运行对应模型类。默认模型列表位于 [chat_models.json](/Users/wangshilong/Downloads/maas-test/chat_models.json)。

基础 chat 套件的设计目标是“每个模型按已知最佳请求形状通过”，而不是强迫所有模型接受同一套最严格语义。因此，`StructuredOutput` 工具路径允许按模型类做最小差异化。

### 模型接口返回特点

| 模型 | 基础 `create` / `stream` | `enable_thinking=false` 行为 | tools 行为 | StructuredOutput 行为 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `kimi-k25` | 正常 | 请求可接受；严格 suppress reasoning 测试当前仍可能返回 `reasoning` 文本 | forced named `tool_choice` 可用，`message.tool_calls` 正常返回 | 强制命名 `StructuredOutput` 工具可复用同一路径 | 即使返回了 `tool_calls`，`finish_reason` 也可能是 `stop` |
| `glm5` | 正常 | 请求可接受，且当前稳定通过严格 suppress reasoning 校验 | forced named `tool_choice` 会返回顶层 `error`；去掉强制 `tool_choice` 后 relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 和 relaxed tools 行为一致 |
| `qwen35` | 默认 thinking 路径偶发超时或流式空 `content`；当前基础文本测试默认使用 `enable_thinking=false` 的稳定路径 | 请求可接受，且当前稳定通过严格 suppress reasoning 校验 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 默认 thinking 打开时不在稳定 passing path |
| `minimax-m25` | 正常 | 请求可接受；严格 suppress reasoning 测试当前仍可能返回 `reasoning` 文本 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 复用同一套工具调用最佳路径 |
| `minimax-m21` | 正常 | 请求可接受；严格 suppress reasoning 测试当前仍可能返回 `reasoning` 文本 | forced named `tool_choice` 返回 `500 upstream_error`；relaxed tools 可用 | 更适合 `tool_choice="auto"` 的 StructuredOutput 工具调用 | 行为基本与 `minimax-m25` 一致 |

## Context Length 套件

context length 套件位于 [tests/test_context_length.py](/Users/wangshilong/Downloads/maas-test/tests/test_context_length.py)。这个文件是默认主套件的一部分，不再是独立脚本。它会对当前选中的每个模型执行一次“先指数扩边、再二分收敛”的上下文边界探测。

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

SDK smoke 套件位于 [tests/test_chat_sdk_smoke.py](/Users/wangshilong/Downloads/maas-test/tests/test_chat_sdk_smoke.py)，默认参与主执行路径；旧的 `--run-sdk-smoke` 仅作为兼容参数保留，不再影响收集结果。

### 1. SDK 基础调用

`test_sdk_create_returns_non_empty_assistant_message`

验证官方 Python SDK 最基本的 `chat.completions.create()` 是否还能成功接入当前服务。

### 2. SDK 原始流式调用

`test_sdk_stream_true_yields_non_empty_text`

验证官方 Python SDK 的 `create(stream=True)` 最基本流式接入是否正常。

## Tool Calling 套件

tool calling 套件位于 [tests/test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/tests/test_tool_calling.py)，默认只保留一个数据集驱动入口：`test_dataset_driven_tool_calling_case`。

### 执行方式

- 请求装载自 [tests/fixtures/k2/tool_calling_subset.jsonl](/Users/wangshilong/Downloads/maas-test/tests/fixtures/k2/tool_calling_subset.jsonl)
- 发送逻辑直接复用 [k2_verifier/core.py](/Users/wangshilong/Downloads/maas-test/k2_verifier/core.py) 的 `ToolCallsValidator`
- 调用面与上游 K2 verifier 保持一致：使用 OpenAI Python SDK，底层连接使用 `httpx.AsyncClient`
- 默认仍属于主套件，不需要额外 marker

### 默认断言

`test_dataset_driven_tool_calling_case`

- 每条样本只校验稳定协议信号，不再校验最终 assistant 文本内容
- 统一断言请求执行成功
- 若样本预期触发工具调用，则断言 `message.tool_calls` 存在，且每个 `function.arguments` 都能通过对应 JSON Schema；即使后端把 `finish_reason` 返回成 `stop` 也按默认 pytest passing path 处理
- 若样本预期是普通结束，则只断言 `finish_reason == "stop"` 且响应结构合法
- 严格的 `finish_reason == "tool_calls"` 口径仍保留给 `python -m k2_verifier.cli` / [k2_verifier/core.py](/Users/wangshilong/Downloads/maas-test/k2_verifier/core.py) 的 K2 verifier 汇总统计

### 当前默认子集覆盖

- 单工具非流式 tool call
- 单工具流式 tool call
- 同一条 assistant 消息中的重复同名 tool call
- 带 `assistant/tool` 历史消息的多轮请求
- 嵌套数组/对象参数的复杂 schema tool call

当前子集会在数据文件元数据里声明不在稳定 passing path 的模型组合；这些组合会在收集阶段直接裁剪，不进入默认结果。比如“重复同名 tool call”目前不会对 `kimi-k25`、`qwen35` 和 `minimax-m21` 执行。

如果某个用例失败，可以直接打开 [test_failure_artifacts](/Users/wangshilong/Downloads/maas-test/test_failure_artifacts) 里的最新归档文件复盘请求和响应细节。
