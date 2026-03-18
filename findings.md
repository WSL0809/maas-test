# Findings

- `B6` 在 [测试功能点.md](/Users/wangshilong/Downloads/maas-test/测试功能点.md) 中定义为“单次回复中并行调用多个工具”。
- 现有 [tests/test_tool_calling.py](/Users/wangshilong/Downloads/maas-test/tests/test_tool_calling.py) 只断言 `finish_reason` 与 `tool_calls_valid`，无法区分“单工具”与“多工具并行”。
- 现有 [tests/fixtures/k2/tool_calling_subset.jsonl](/Users/wangshilong/Downloads/maas-test/tests/fixtures/k2/tool_calling_subset.jsonl) 只有单工具、同名重复工具、多轮历史、复杂 schema 等 case，没有“多个不同工具同消息”。
- verifier 在 [k2_verifier/core.py](/Users/wangshilong/Downloads/maas-test/k2_verifier/core.py) 已能拿到完整 `tool_calls` 列表，扩展精确断言的改动面较小。
- 已补充 `expected_tool_call_names` metadata，并在 verifier 结果中输出 `tool_call_names` / `tool_call_names_match`，可以无序校验工具名称多重集。
- 已新增 B6 fixture：`parallel_distinct_tool_calls`，要求同一条 assistant 消息中同时调用 `lookup_weather` 与 `lookup_local_time`。
- live 复测结果：`glm5`、`kimi-k25`、`minimax-m25` 通过；`qwen35` 因 reasoning 占满 completion 长度未产出 `tool_calls`；`minimax-m21` 输出文本/XML 形态的伪工具调用并被截断。
- 已将 `parallel_distinct_tool_calls` 的稳定路径过滤为跳过 `qwen35` 与 `minimax-m21`，并回填到 README、`test_run.md`、`测试功能点.md`。
