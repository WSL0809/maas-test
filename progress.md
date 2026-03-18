# Progress

- 2026-03-18: Initialized planning files for finishing the easy items in section A.
- 2026-03-18: Confirmed current state: A1 and A2 are complete; A3-A12 remain mostly untested.
- 2026-03-18: Added low-risk A tests for system prompt priority, max token limits, multilingual output, and special token preservation.
- 2026-03-18: Verified A4/A5 via existing create/stream tests and verified A8/A11/A12 across all current models.
- 2026-03-18: A3 is partial only: `minimax-m25` passed while `minimax-m21` failed, so the grouped Minimax column is marked `⚠️`.
- 2026-03-18: Ran low-risk B coverage already present in the suite. `enable_thinking=false` create/stream and `StructuredOutput` passed on the current matrix, with `kimi-k25` / Minimax still only partial for the "no reasoning text" expectation.
- 2026-03-18: Re-ran dataset-driven tool calling and kept only directly supported conclusions: `history_with_tool_messages` is stable across all models, but current B-state backfill only uses `single_tool_nonstream`, where only `kimi-k25` passes and the other models reproduce backend-side `500` / `upstream_error`.
- 2026-03-18: Added explicit B1 thinking-mode tests in `tests/test_chat.py` for both create and stream. Current acceptance rule is `message.reasoning` non-empty plus non-empty streamed reasoning deltas.
- 2026-03-18: Verified B1 across all current models with `uv run pytest -q tests/test_chat.py -k 'thinking_enabled'`: `10 passed`, so the B1 matrix row is now marked `✅` for all models.
