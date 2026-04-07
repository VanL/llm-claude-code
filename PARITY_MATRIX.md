# Parity Matrix: `llm-claude-code` vs `llm-anthropic`

Last updated: March 3, 2026

Reference baselines:
- `llm` 0.28
- `llm-anthropic` 0.24
- `claude-agent-sdk` 0.1.44

Status legend:
- `SUPPORTED`: behavior intentionally matches `llm-anthropic` or SDK-native equivalent.
- `PARTIAL`: supported with caveats or degraded fallback.
- `UNSUPPORTED`: no safe equivalent in Claude SDK today.

## Core Runtime

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| Model registration + aliases | Anthropic-style stable aliases | SUPPORTED | `llm models` exposes sync + async `claude-code-*` models across the Claude 4.0/4.1/4.5/4.6 families plus Haiku 4.5 aliases. |
| Sync/async execution parity | Same behavior between sync and async pathways | SUPPORTED | Shared execution logic; tests verify identical usage/tool behavior. |
| Completion detection | End only on definitive terminal event | SUPPORTED | Treat `ResultMessage` as terminal completion signal. |
| Response metadata | Preserve raw provider response for debugging | SUPPORTED | `response.response_json` includes serialized SDK message stream. |

## Prompt / Conversation Construction

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| Conversation history | Preserve user + assistant turns + tool artifacts | PARTIAL | Structured message objects built; fallback text transcript used when SDK input mode cannot safely replay assistant turns. |
| Attachments (image/PDF/url/base64) | Native block pass-through | PARTIAL | Attachment blocks are built and sent in stream-json mode for user turns; fallback transcript used for complex multi-turn replay. |
| Prefill + hide_prefill | Match `llm-anthropic` semantics | PARTIAL | Prefill is included in prompt context; visible output obeys `hide_prefill`. |

## Options / Controls

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| `system`, `append_system_prompt`, `cwd`, `resume`, `continue_conversation` | SDK-native mapping | SUPPORTED | Option values map directly to SDK options when available. |
| `thinking`, `thinking_budget`, `max_thinking_tokens`, `thinking_effort` | Anthropic-style thinking knobs | PARTIAL | Mapped to SDK `thinking`, `max_thinking_tokens`, `effort` where available, including adaptive thinking for 4.6 models and budget-only thinking activation. |
| `user_id` | Anthropic metadata parity | PARTIAL | Mapped to SDK `user` when available. |
| `temperature`, `top_p`, `top_k`, `max_tokens`, `stop_sequences`, `cache` | Sampling/cache parity | UNSUPPORTED | No direct SDK equivalents; values are tracked as ignored and surfaced in response metadata. |
| `web_search*` | Anthropic web-search option parity | PARTIAL | `web_search=True` maps to enabling `WebSearch` tool; advanced search constraints are rejected with explicit errors. |

## Tools / Structured Output

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| Tool call extraction | Add `llm.ToolCall` from model tool uses | SUPPORTED | `ToolUseBlock` becomes `response.tool_calls()`. |
| Python callable tool loop parity | Full `llm-anthropic` tool-calling loop | PARTIAL | Tool uses are surfaced; full custom function schema/tool execution parity is not guaranteed with Claude Code built-in tool model. |
| Schema/JSON output | `prompt.schema` and JSON stream support | PARTIAL | Uses SDK `output_format` when available; falls back to explicit schema instruction otherwise. |
| Partial JSON streaming | Streaming partial JSON for schema calls | PARTIAL | Stream events parsed for JSON deltas when available and opted in. |

## Usage / Accounting

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| Input/output token accounting | Use provider numbers when present | SUPPORTED | Token usage read from `ResultMessage.usage`, fallback estimate only when missing. |
| Usage details | Preserve extra usage metadata | SUPPORTED | Extra usage keys exposed in `response.token_details`. |

## Tests and QA

| Area | Target parity | Status | Acceptance criteria |
|---|---|---|---|
| Unit tests for regressions | Cover completion/tool/schema/usage failures | SUPPORTED | Deterministic tests pass without external API. |
| Integration tests with real Claude CLI/API | End-to-end parity validation | PARTIAL | Deferred to optional integration suite; not required for offline CI. |

---

## Implementation Checklist

- [x] Shared sync/async execution core.
- [x] `ResultMessage` terminal completion handling.
- [x] Structured response serialization into `response_json`.
- [x] Anthropic-style message builder with attachment/tool blocks.
- [x] Schema to SDK `output_format` mapping with fallback instruction mode.
- [x] Option compatibility layer with explicit unsupported reporting.
- [x] Web-search shim + explicit rejection for unsupported sub-options.
- [x] Expand model catalog to mirror latest Anthropic lineup naming breadth.
- [ ] Add optional live integration test suite (guarded by env vars).
- [ ] Implement stronger custom Python tool parity semantics where SDK allows.
