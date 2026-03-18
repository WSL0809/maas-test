# Task Plan

## Goal

Complete the easy items in section A of `测试功能点.md`, skip unstable or high-effort items for now, and record outcomes with emoji statuses.

## Phases

| Phase | Status | Notes |
|---|---|---|
| Identify easy A3-A12 items and existing coverage | complete | A4/A5 reused existing coverage; A3/A8/A11/A12 were added as low-risk tests. |
| Implement missing low-risk tests in `tests/test_chat.py` | complete | README and CSV descriptions updated together. |
| Run the selected A tests across model matrix | complete | A3/A4/A5/A8/A11/A12 verified; A6/A7/A9/A10 deferred. |
| Update `测试功能点.md` statuses | complete | Added `⚠️` for partial pass in grouped Minimax column. |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `minimax-m21` failed A3 system prompt priority | 1 | Kept A3 as partial pass for grouped Minimax column and deferred deeper debugging. |
