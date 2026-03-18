# Progress

## 2026-03-18

- 切换到 B7 任务，确认现有 dataset-driven verifier 不会自动执行多步 tool loop。
- 在 `tests/test_tool_calling.py` 中新增 `test_multi_step_tool_chain_round_trip`，并实现最小的本地 tool 执行回填逻辑。
- 用 `fetch_seed_word -> uppercase_word -> decorate_word -> [STONE]` 这条 3 步链验证“工具结果作为下一步输入”。
- 定向 live 复测结果：`glm5`、`minimax-m21`、`minimax-m25` 通过；`qwen35` 与 `kimi-k25` 当前不稳定。
- 已完成 B7 文档回填：README、`test_run.md`、`测试功能点.md`、CSV 描述注册。
- 补跑整份 `tests/test_tool_calling.py` 时发现多个旧 case 在现网后端上漂移失败；该问题已记录，但本轮未扩散修复。
