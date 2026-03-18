# Findings

- A1 is covered by `test_create_returns_non_empty_assistant_message` and already passes for all current models.
- A2 required a new multi-turn test and now passes for all current models.
- A4 already maps cleanly to `test_stream_sse_emits_content_and_done`.
- A5 can reuse the non-stream create path from A1.
- A3, A8, A11, and A12 appear implementable with stable prompt-and-shape assertions.
- A6, A7, A9, and A10 are more likely to be backend-sensitive or flaky and should be deferred unless the easy set is exhausted.
- A3 now passes on `kimi-k25`, `glm5`, `qwen35`, and `minimax-m25`, but `minimax-m21` still follows the user instruction in the conflict case.
- A8 passes after asserting the token cap via `usage.completion_tokens`, even when some models may return empty content under tight limits.
- A11 needed punctuation normalization because `qwen35` preferred full-width commas.
- A12 passes for all current models with containment-style assertions on emoji / HTML / code / math symbols.
