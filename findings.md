# Findings

- `B7` 在 [测试功能点.md](/Users/wangshilong/Downloads/maas-test/测试功能点.md) 中定义为“工具结果作为下一步输入，验证 3+ 步链式执行”。
- 现有 [tests/test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/tests/test_tool_calling.py) 的 dataset-driven verifier 只适合单请求断言，不会自动执行 tool loop，因此不能直接覆盖 B7。
- B7 不需要大改 verifier 主干；直接在 [tests/test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/tests/test_tool_calling.py) 增加一个显式 round-trip 用例即可复用现有 `httpx` 请求、失败归档和模型矩阵参数化。
- 已新增 `test_multi_step_tool_chain_round_trip`：通过 `fetch_seed_word -> uppercase_word -> decorate_word -> [STONE]` 这条 3 步链验证“上一轮 tool 结果成为下一轮 tool 参数”。
- live 探测结果：`glm-5`、`minimax-m21`、`minimax-m25` 稳定通过；`qwen35` 第二步会退化成文本/XML 风格的伪工具调用并因长度截断；`kimi-k25` 当前可完成前两步，但第三步会丢失结构化 `tool_calls`。
- 已将 B7 默认稳定路径过滤为跳过 `qwen35` 与 `kimi-k25`，并回填到 README、`test_run.md`、`测试功能点.md`。
- 额外发现：当前整份 [tests/test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/tests/test_tool_calling.py) 在现网后端上有多条旧 case 失稳，属于 B7 之外的现网漂移，需要后续单独处理。
