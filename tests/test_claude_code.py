import llm
import pytest
from pydantic import BaseModel

import llm_claude_code as plugin


MODEL_ID = "claude-code-sonnet-4-20250514"

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\xa6\x00\x00\x01\x1a"
    b"\x02\x03\x00\x00\x00\xe6\x99\xc4^\x00\x00\x00\tPLTE\xff\xff\xff"
    b"\x00\xff\x00\xfe\x01\x00\x12t\x01J\x00\x00\x00GIDATx\xda\xed\xd81\x11"
    b"\x000\x08\xc0\xc0.]\xea\xaf&Q\x89\x04V\xe0>\xf3+\xc8\x91Z\xf4\xa2\x08EQ\x14E"
    b"Q\x14EQ\x14EQ\xd4B\x91$I3\xbb\xbf\x08EQ\x14EQ\x14EQ\x14E\xd1\xa5"
    b"\xd4\x17\x91\xc6\x95\x05\x15\x0f\x9f\xc5\t\x9f\xa4\x00\x00\x00\x00IEND\xaeB`"
    b"\x82"
)


def model_config_for(model_id):
    return {
        k: v for k, v in plugin.MODEL_CONFIGS[model_id].items() if k != "aliases"
    }


class Dog(BaseModel):
    name: str
    age: int


def make_model(model_id=MODEL_ID):
    model = plugin.ClaudeCodeMessages(model_id, model_config_for(model_id))
    model.key = "sk-test"
    return model


def make_result(
    *,
    result="done",
    usage=None,
    is_error=False,
    structured_output=None,
):
    return plugin.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="session-1",
        total_cost_usd=0.0001,
        usage=usage or {"input_tokens": 3, "output_tokens": 4},
        result=result,
        structured_output=structured_output,
    )


def test_sync_response_collects_usage(monkeypatch):
    async def fake_query(*, prompt, options):
        yield plugin.AssistantMessage(
            content=[plugin.TextBlock(text="Hello world")],
            model="claude-sonnet-4",
        )
        yield make_result(result="Hello world", usage={"input_tokens": 11, "output_tokens": 7})

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt("Say hello", stream=False)

    assert response.text() == "Hello world"
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.resolved_model == "claude-sonnet-4"
    assert response.response_json["sdk"] in ("claude-agent-sdk", "claude-code-sdk")


def test_result_fallback_if_no_assistant_text(monkeypatch):
    async def fake_query(*, prompt, options):
        yield make_result(result="final only")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt("Give final answer", stream=False)

    assert response.text() == "final only"


def test_partial_stream_events(monkeypatch):
    async def fake_query(*, prompt, options):
        yield plugin.StreamEvent(
            uuid="u1",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Par"},
            },
        )
        yield plugin.StreamEvent(
            uuid="u2",
            session_id="session-1",
            event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "tial"},
            },
        )
        yield make_result(result=None)

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt(
        "Stream partial",
        include_partial_messages=True,
        stream=True,
    )
    assert response.text() == "Partial"


def test_tool_use_block_creates_tool_call(monkeypatch):
    async def fake_query(*, prompt, options):
        yield plugin.AssistantMessage(
            content=[
                plugin.ToolUseBlock(
                    id="tool-1",
                    name="Read",
                    input={"file_path": "README.md"},
                )
            ],
            model="claude-sonnet-4",
        )
        yield make_result(result="done")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt("Use tool", stream=False)
    tool_calls = response.tool_calls()

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id == "tool-1"
    assert tool_calls[0].name == "Read"
    assert tool_calls[0].arguments == {"file_path": "README.md"}


def test_permission_mode_normalization(monkeypatch):
    captured = {}

    async def fake_query(*, prompt, options):
        captured["options"] = options
        yield make_result(result="ok")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    model.prompt("mode test", permission_mode="ask", stream=False).text()

    if plugin.SDK_NAME == "claude-agent-sdk":
        assert captured["options"].permission_mode == "default"
    else:
        assert captured["options"].permission_mode == "ask"


def test_register_models():
    seen = []
    alias_map = {}

    def register(sync_model, async_model, aliases=None):
        seen.append((sync_model.model_id, async_model.model_id, aliases))
        for alias in aliases or ():
            alias_map[alias] = sync_model.model_id

    plugin.register_models(register)

    registered_ids = [entry[0] for entry in seen]
    assert "anthropic/claude-code-opus-4-20250514" in registered_ids
    assert "anthropic/claude-code-sonnet-4-20250514" in registered_ids
    assert "anthropic/claude-code-opus-4-6" in registered_ids
    assert "anthropic/claude-code-sonnet-4-6" in registered_ids
    assert "anthropic/claude-code-haiku-4-5-20251001" in registered_ids
    assert alias_map["claude-code-opus"] == "anthropic/claude-code-opus-4-6"
    assert alias_map["claude-code-sonnet"] == "anthropic/claude-code-sonnet-4-6"
    assert alias_map["claude-code-haiku"] == "anthropic/claude-code-haiku-4-5-20251001"


def test_schema_sets_output_format_when_sdk_supports_it(monkeypatch):
    captured = {}

    async def fake_query(*, prompt, options):
        captured["options"] = options
        yield make_result(result=None, structured_output={"name": "Tarantula Bob", "age": 6})

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt("Return a dog", schema=Dog, stream=False)
    text = response.text()
    assert '"name": "Tarantula Bob"' in text

    if hasattr(captured["options"], "output_format"):
        output_format = getattr(captured["options"], "output_format", None)
        if output_format:
            assert output_format["type"] == "json_schema"


def test_stream_json_mode_with_single_user_prompt(monkeypatch):
    captured = {"events": None}

    async def fake_query(*, prompt, options):
        if not isinstance(prompt, str):
            events = []
            async for event in prompt:
                events.append(event)
            captured["events"] = events
        yield plugin.AssistantMessage(
            content=[plugin.TextBlock(text="OK")],
            model="claude-sonnet-4",
        )
        yield make_result(result="OK")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt(
        "Describe image",
        attachments=[llm.Attachment(content=TINY_PNG)],
        stream=False,
    )
    assert response.text() == "OK"
    assert response.response_json["prompt_mode"] == "stream_json"
    assert captured["events"]
    first_content = captured["events"][0]["message"]["content"]
    assert first_content[0]["type"] == "image"


def test_fallback_prompt_mode_for_conversation_history(monkeypatch):
    captured_modes = []

    async def fake_query(*, prompt, options):
        captured_modes.append("string" if isinstance(prompt, str) else "stream")
        if not isinstance(prompt, str):
            async for _ in prompt:
                pass
        yield plugin.AssistantMessage(
            content=[plugin.TextBlock(text="answer")],
            model="claude-sonnet-4",
        )
        yield make_result(result="answer")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    conversation = model.conversation()
    response1 = conversation.prompt("first", stream=False)
    assert response1.text() == "answer"
    response2 = conversation.prompt("second", stream=False)
    assert response2.text() == "answer"
    assert captured_modes == ["stream", "string"]
    assert response2.response_json["prompt_mode"] == "fallback_text"


def test_web_search_shim_adds_allowed_tool(monkeypatch):
    captured = {}

    async def fake_query(*, prompt, options):
        captured["allowed_tools"] = getattr(options, "allowed_tools", [])
        if not isinstance(prompt, str):
            async for _ in prompt:
                pass
        yield make_result(result="ok")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    assert model.prompt("search test", web_search=True, stream=False).text() == "ok"
    assert "WebSearch" in captured["allowed_tools"]


def test_web_search_advanced_options_raise_error():
    model = make_model()
    response = model.prompt(
        "search test",
        web_search=True,
        web_search_allowed_domains=["example.com"],
        stream=False,
    )
    with pytest.raises(ValueError):
        response.text()


def test_ignored_sampling_options_are_reported(monkeypatch):
    async def fake_query(*, prompt, options):
        if not isinstance(prompt, str):
            async for _ in prompt:
                pass
        yield make_result(result="ok")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model()
    response = model.prompt(
        "sample",
        temperature=0.3,
        top_k=10,
        stop_sequences=["!"],
        stream=False,
    )
    assert response.text() == "ok"
    ignored = set(response.response_json["ignored_options"])
    assert {"temperature", "top_k", "stop_sequences"} <= ignored


def test_thinking_budget_enables_thinking(monkeypatch):
    captured = {}

    async def fake_query(*, prompt, options):
        captured["options"] = options
        if not isinstance(prompt, str):
            async for _ in prompt:
                pass
        yield make_result(result="ok")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model("claude-code-sonnet-4-5")
    response = model.prompt("think", thinking_budget=2048, stream=False)

    assert response.text() == "ok"
    assert captured["options"].thinking == {"type": "enabled", "budget_tokens": 2048}


def test_adaptive_thinking_used_for_46_models(monkeypatch):
    captured = {}

    async def fake_query(*, prompt, options):
        captured["options"] = options
        if not isinstance(prompt, str):
            async for _ in prompt:
                pass
        yield make_result(result="ok")

    monkeypatch.setattr(plugin, "query", fake_query)

    model = make_model("claude-code-sonnet-4-6")
    response = model.prompt("think", thinking=True, stream=False)

    assert response.text() == "ok"
    assert captured["options"].thinking == {"type": "adaptive"}


def test_prefill_rejected_for_46_models():
    model = make_model("claude-code-sonnet-4-6")
    response = model.prompt("hello", prefill="{", stream=False)

    with pytest.raises(ValueError, match="Prefilling assistant messages is not supported"):
        response.text()


def test_max_effort_restricted_to_opus_46():
    model = make_model("claude-code-sonnet-4-6")
    response = model.prompt("hello", thinking_effort="max", stream=False)

    with pytest.raises(ValueError, match="thinking_effort='max' is only supported"):
        response.text()
