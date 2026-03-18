# Task Plan

## Goal

继续完成 B6「工具调用-并行调用」场景，确保测试能精确验证“单次 assistant 回复里包含多个工具调用”，并将结果回填到仓库文档。

## Phases

| Phase | Status | Notes |
|---|---|---|
| Inspect current B6 definition and verifier capabilities | complete | 已确认当前只校验 tool call schema 合法，不能精确断言多工具并行调用 |
| Extend verifier and fixtures to express multi-tool parallel-call expectations | complete | 已补 metadata 字段、定向 fixture 与断言 |
| Run targeted tests for B6 and update docs/status | in_progress | 已完成首轮 live pytest 和文档回填，正在做最终回归 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| None | 0 | N/A |
