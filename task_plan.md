# Task Plan

## Goal

完成 B7「工具调用-多步链式」场景，补一条可执行的 3 步 tool loop 回填测试，并将稳定 passing path 回填到仓库文档。

## Phases

| Phase | Status | Notes |
|---|---|---|
| Inspect current B7 definition and reusable tool-calling infrastructure | complete | 已确认现有 dataset-driven verifier 适合单请求校验，B7 需要显式 tool loop round-trip 帮助函数 |
| Implement a minimal 3-step chain tool-loop test | complete | 已在 `tests/test_tool_calling.py` 新增 `test_multi_step_tool_chain_round_trip` 与链式本地工具执行 |
| Probe model matrix and determine the stable B7 passing path | complete | `glm5`、`minimax-m21`、`minimax-m25` 通过；`qwen35`、`kimi-k25` 当前不稳定 |
| Update docs and close out the B7 task | complete | 已回填 README、`test_run.md`、`测试功能点.md` 与 CSV 描述 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| Kimi B7 chain stalled on step 3 | 1 | 先尝试 `enable_thinking=false`，无效后改为压缩提示并增大 `max_completion_tokens`；最终记录为当前非稳定路径 |
| Full `tests/test_tool_calling.py` matrix regressed outside B7 scope | 1 | 保留 B7 定向结果，后续需要单独梳理现网 tool-calling 旧 case 漂移 |
