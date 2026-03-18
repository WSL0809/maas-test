# Progress

## 2026-03-18

- 定位 B6 定义与当前状态，确认仍为未回填。
- 检查 tool-calling 套件与 verifier 核心，发现缺少“多工具并行调用”的精确断言与 fixture。
- 开始补充 B6 基础设施与定向验证。
- 已修改 verifier metadata / result 结构，并更新 `tests/test_tool_calling.py` 以断言期望的 tool 名称集合。
- 已新增 `parallel_distinct_tool_calls` fixture，并让 `repeated_same_tool_calls` 也使用精确 tool 名称匹配。
- 已完成 B6 live 探测，识别出稳定通过模型与失败模型的具体失败形态。
- 正在做最终回归：单元测试 + B6 targeted pytest。
