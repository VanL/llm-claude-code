"""
LLM plugin for Claude Code / Claude Agent SDK.
"""

import asyncio
import json
import os
import queue
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import llm

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions as SDKClaudeOptions,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )
    from claude_agent_sdk.types import StreamEvent

    SDK_NAME = "claude-agent-sdk"
except ImportError:
    try:
        from claude_code_sdk import (  # type: ignore[no-redef]
            AssistantMessage,
            ClaudeCodeOptions as SDKClaudeOptions,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )
        from claude_code_sdk.types import StreamEvent  # type: ignore[no-redef]

        SDK_NAME = "claude-code-sdk"
    except ImportError as ex:
        raise ImportError(
            "Install either claude-agent-sdk (recommended) or claude-code-sdk"
        ) from ex


DEFAULT_TEMPERATURE = 1.0
DEFAULT_THINKING_TOKENS = 1024


MODEL_CONFIGS = {
    "claude-code-opus-4-20250514": {
        "claude_model_id": "claude-opus-4-20250514",
        "sdk_model": "claude-opus-4-20250514",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-opus-4",
        ],
    },
    "claude-code-opus-4-1-20250805": {
        "claude_model_id": "claude-opus-4-1-20250805",
        "sdk_model": "claude-opus-4-1-20250805",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-opus-4.1",
        ],
    },
    "claude-code-opus-4-5-20251101": {
        "claude_model_id": "claude-opus-4-5-20251101",
        "sdk_model": "claude-opus-4-5-20251101",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_thinking_effort": True,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-opus-4.5",
        ],
    },
    "claude-code-opus-4-6": {
        "claude_model_id": "claude-opus-4-6",
        "sdk_model": "claude-opus-4-6",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_thinking_effort": True,
        "supports_adaptive_thinking": True,
        "supports_max_thinking_effort": True,
        "supports_assistant_prefill": False,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-opus",
            "claude-code-opus-4.6",
            "claude-code-opus-latest",
        ],
    },
    "claude-code-sonnet-4-20250514": {
        "claude_model_id": "claude-sonnet-4-20250514",
        "sdk_model": "claude-sonnet-4-20250514",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-sonnet-4",
        ],
    },
    "claude-code-sonnet-4-5": {
        "claude_model_id": "claude-sonnet-4-5",
        "sdk_model": "claude-sonnet-4-5",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-sonnet-4.5",
        ],
    },
    "claude-code-sonnet-4-6": {
        "claude_model_id": "claude-sonnet-4-6",
        "sdk_model": "claude-sonnet-4-6",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "supports_thinking_effort": True,
        "supports_adaptive_thinking": True,
        "supports_assistant_prefill": False,
        "supports_web_search": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-sonnet",
            "claude-code-sonnet-4.6",
            "claude-code-sonnet-latest",
        ],
    },
    "claude-code-haiku-4-5-20251001": {
        "claude_model_id": "claude-haiku-4-5-20251001",
        "sdk_model": "claude-haiku-4-5-20251001",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "default_max_tokens": 8192,
        "aliases": [
            "claude-code-haiku",
            "claude-code-haiku-4.5",
            "claude-code-haiku-latest",
        ],
    },
}


def source_for_attachment(attachment):
    if attachment.url:
        return {
            "type": "url",
            "url": attachment.url,
        }
    return {
        "data": attachment.base64_content(),
        "media_type": attachment.resolve_type(),
        "type": "base64",
    }


def _parse_optional_list(value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return None


def _parse_optional_object(value: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as ex:
            raise ValueError("Expected a JSON object") from ex
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected a JSON object")


def _parse_stop_sequences(value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    parsed = _parse_optional_list(value)
    if parsed:
        return parsed
    return None


def _schema_to_dict(schema: Any) -> Optional[Dict[str, Any]]:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    raise ValueError("Schema must be a JSON schema dict or a Pydantic model class")


def _normalize_permission_mode(raw_mode: Optional[str]) -> Optional[str]:
    if raw_mode is None:
        return None

    mode = raw_mode.strip()
    if not mode:
        return None

    if SDK_NAME == "claude-agent-sdk":
        mapping = {
            "ask": "default",
            "acceptAll": "bypassPermissions",
        }
    else:
        mapping = {
            "default": "ask",
            "bypassPermissions": "acceptAll",
        }
    return mapping.get(mode, mode)


class ClaudeCodeOptions(llm.Options):
    # SDK-native options
    system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    tools: Optional[Union[str, List[str]]] = None
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    permission_mode: Optional[str] = None
    permission_prompt_tool_name: Optional[str] = None
    cwd: Optional[str] = None
    add_dirs: Optional[Union[str, List[str]]] = None
    continue_conversation: Optional[bool] = None
    resume: Optional[str] = None
    include_partial_messages: bool = False
    setting_sources: Optional[Union[str, List[str]]] = None
    max_budget_usd: Optional[float] = None
    fallback_model: Optional[str] = None
    betas: Optional[Union[str, List[str]]] = None
    settings: Optional[str] = None
    output_format: Optional[Union[str, Dict[str, Any]]] = None

    # Claude thinking controls
    thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    max_thinking_tokens: Optional[int] = None
    effort: Optional[str] = None
    thinking_effort: Optional[str] = None

    # llm-anthropic compatibility options
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[Union[str, List[str]]] = None
    user_id: Optional[str] = None
    prefill: Optional[str] = None
    hide_prefill: bool = False
    cache: Optional[bool] = None
    web_search: bool = False
    web_search_max_uses: Optional[int] = None
    web_search_allowed_domains: Optional[Union[str, List[str]]] = None
    web_search_blocked_domains: Optional[Union[str, List[str]]] = None
    web_search_location: Optional[Union[str, Dict[str, Any]]] = None


class _Shared:
    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True
    supports_schema = True
    supports_tools = True
    Options = ClaudeCodeOptions

    def __init__(self, model_id: str, config: dict):
        self.model_id = "anthropic/" + model_id
        self.claude_model_id = config.get(
            "claude_model_id",
            model_id.replace("claude-code-", "claude-"),
        )
        self.sdk_model = config.get("sdk_model")
        self.supports_images = config.get("supports_images", False)
        self.supports_pdf = config.get("supports_pdf", False)
        self.supports_thinking = config.get("supports_thinking", False)
        self.supports_thinking_effort = config.get("supports_thinking_effort", False)
        self.supports_adaptive_thinking = config.get(
            "supports_adaptive_thinking",
            False,
        )
        self.supports_max_thinking_effort = config.get(
            "supports_max_thinking_effort",
            False,
        )
        self.supports_assistant_prefill = config.get(
            "supports_assistant_prefill",
            True,
        )
        self.supports_web_search = config.get("supports_web_search", False)
        self.default_max_tokens = config.get("default_max_tokens", 4096)
        self._api_key = None

        self.attachment_types = set()
        if self.supports_images:
            self.attachment_types.update(
                {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }
            )
        if self.supports_pdf:
            self.attachment_types.add("application/pdf")

    def get_key(self, key=None):
        if key:
            return key
        from_env = os.environ.get(self.key_env_var)
        if from_env:
            return from_env
        model_key = getattr(self, "key", None)
        if model_key:
            return model_key
        return self._api_key

    def prompt_blocks(self):
        remove_role = getattr(llm, "remove_role_block", None)
        if remove_role is not None:
            yield remove_role()
        multi_binary_attachments = getattr(llm, "multi_binary_attachments_block", None)
        if multi_binary_attachments is not None:
            yield multi_binary_attachments()

    def _prompt_options(self, prompt: llm.Prompt) -> ClaudeCodeOptions:
        options = getattr(prompt, "options", None)
        if options is None:
            return ClaudeCodeOptions()
        return options

    def _prefill_text(self, prompt: llm.Prompt) -> str:
        options = self._prompt_options(prompt)
        if options.prefill and not options.hide_prefill:
            return options.prefill
        return ""

    def _response_text(self, response_obj: Any) -> str:
        try:
            if hasattr(response_obj, "text_or_raise"):
                return response_obj.text_or_raise()
        except Exception:
            return ""
        try:
            if hasattr(response_obj, "text"):
                return response_obj.text()
        except Exception:
            return ""
        return ""

    def _response_tool_calls(self, response_obj: Any) -> List[Any]:
        try:
            if hasattr(response_obj, "tool_calls_or_raise"):
                return response_obj.tool_calls_or_raise()
        except Exception:
            return []
        return []

    def _build_messages(self, prompt: llm.Prompt, conversation=None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []

        if conversation:
            for previous_response in conversation.responses:
                user_content: List[Dict[str, Any]] = []

                response_attachments = getattr(previous_response, "attachments", None)
                if response_attachments:
                    for attachment in response_attachments:
                        attachment_type = (
                            "document"
                            if attachment.resolve_type() == "application/pdf"
                            else "image"
                        )
                        user_content.append(
                            {
                                "type": attachment_type,
                                "source": source_for_attachment(attachment),
                            }
                        )

                previous_prompt_text = getattr(previous_response.prompt, "prompt", None)
                if previous_prompt_text:
                    user_content.append(
                        {
                            "type": "text",
                            "text": previous_prompt_text,
                        }
                    )

                response_tool_results = getattr(previous_response.prompt, "tool_results", None)
                if response_tool_results:
                    for tool_result in response_tool_results:
                        user_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result.tool_call_id,
                                "content": tool_result.output,
                            }
                        )

                if user_content:
                    messages.append({"role": "user", "content": user_content})

                assistant_content: List[Dict[str, Any]] = []
                text_content = self._response_text(previous_response)
                if text_content:
                    assistant_content.append({"type": "text", "text": text_content})

                for tool_call in self._response_tool_calls(previous_response):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.tool_call_id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    )

                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

        user_content: List[Dict[str, Any]] = []

        if prompt.attachments:
            for attachment in prompt.attachments:
                attachment_type = (
                    "document"
                    if attachment.resolve_type() == "application/pdf"
                    else "image"
                )
                user_content.append(
                    {
                        "type": attachment_type,
                        "source": source_for_attachment(attachment),
                    }
                )

            options = self._prompt_options(prompt)
            if options.cache and user_content:
                user_content[-1]["cache_control"] = {"type": "ephemeral"}

        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                user_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": tool_result.output,
                    }
                )

        if prompt.prompt:
            user_content.append({"type": "text", "text": prompt.prompt})

        if user_content:
            messages.append({"role": "user", "content": user_content})

        options = self._prompt_options(prompt)
        if options.prefill:
            if not self.supports_assistant_prefill:
                raise ValueError(
                    f"Prefilling assistant messages is not supported by {self.claude_model_id}. "
                    "Use structured outputs or system prompt instructions instead."
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": options.prefill}],
                }
            )

        return messages

    def _content_block_to_text(self, block: Dict[str, Any]) -> str:
        block_type = block.get("type")
        if block_type == "text":
            return block.get("text", "")
        if block_type == "tool_result":
            return "Tool result ({tool_use_id}): {content}".format(
                tool_use_id=block.get("tool_use_id", "unknown"),
                content=block.get("content", ""),
            )
        if block_type == "tool_use":
            return "Tool call {name}: {arguments}".format(
                name=block.get("name", "unknown"),
                arguments=json.dumps(block.get("input", {})),
            )
        if block_type in ("image", "document"):
            source = block.get("source", {})
            if isinstance(source, dict):
                if source.get("type") == "url":
                    location = source.get("url", "attachment")
                else:
                    location = source.get("media_type", "binary data")
            else:
                location = "attachment"
            label = "Image" if block_type == "image" else "Document"
            return f"[{label} attachment: {location}]"
        return str(block)

    def _messages_to_fallback_prompt(
        self,
        messages: List[Dict[str, Any]],
        schema_instruction: Optional[str],
    ) -> str:
        parts: List[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            role_label = "Human" if role == "user" else "Assistant"

            if isinstance(content, str):
                parts.append(f"{role_label}: {content}")
                continue

            lines = [self._content_block_to_text(block) for block in content]
            lines = [line for line in lines if line]
            if lines:
                parts.append(f"{role_label}: " + "\n".join(lines))

        if schema_instruction:
            parts.append(schema_instruction)

        return "\n\n".join(parts)

    def _build_query_prompt(
        self,
        messages: List[Dict[str, Any]],
        schema_instruction: Optional[str],
    ) -> tuple[Union[str, AsyncIterator[Dict[str, Any]]], str]:
        has_assistant_history = any(message.get("role") != "user" for message in messages)

        if has_assistant_history:
            return (
                self._messages_to_fallback_prompt(messages, schema_instruction),
                "fallback_text",
            )

        async def _input_stream() -> AsyncIterator[Dict[str, Any]]:
            for message in messages:
                yield {
                    "type": "user",
                    "session_id": "",
                    "message": message,
                    "parent_tool_use_id": None,
                }

        return _input_stream(), "stream_json"

    def _build_options(
        self,
        prompt: llm.Prompt,
    ) -> tuple[SDKClaudeOptions, List[str], Optional[str]]:
        options = self._prompt_options(prompt)
        sdk_fields = set(getattr(SDKClaudeOptions, "__dataclass_fields__", {}).keys())
        sdk_options: Dict[str, Any] = {}
        ignored_options: List[str] = []
        thinking_effort = options.effort or options.thinking_effort
        thinking_requested = bool(options.thinking) or options.thinking_budget is not None

        if options.top_p is not None and options.temperature not in (None, DEFAULT_TEMPERATURE):
            raise ValueError("Only one of temperature and top_p can be set")

        if options.effort and options.thinking_effort and options.effort != options.thinking_effort:
            raise ValueError("Use either effort or thinking_effort, not both")

        if options.thinking_effort and not self.supports_thinking_effort:
            raise ValueError(
                f"thinking_effort is not supported by {self.claude_model_id}"
            )

        if thinking_effort and not options.thinking and not self.supports_adaptive_thinking:
            thinking_requested = True

        if thinking_effort == "max" and not self.supports_max_thinking_effort:
            raise ValueError(
                "thinking_effort='max' is only supported by claude-opus-4-6"
            )

        if "system_prompt" in sdk_fields and prompt.system:
            sdk_options["system_prompt"] = prompt.system
        elif "system_prompt" in sdk_fields and options.system_prompt:
            sdk_options["system_prompt"] = options.system_prompt

        if "append_system_prompt" in sdk_fields and options.append_system_prompt:
            sdk_options["append_system_prompt"] = options.append_system_prompt

        if "model" in sdk_fields:
            sdk_options["model"] = self.sdk_model or self.claude_model_id

        if "max_turns" in sdk_fields:
            sdk_options["max_turns"] = options.max_turns if options.max_turns is not None else 1

        parsed_tools = _parse_optional_list(options.tools)
        if prompt.tools:
            parsed_tools = parsed_tools or []
            for tool in prompt.tools:
                name = getattr(tool, "name", None)
                if name:
                    parsed_tools.append(name)

        if "tools" in sdk_fields and parsed_tools:
            sdk_options["tools"] = parsed_tools
        elif parsed_tools:
            ignored_options.append("tools")

        if "allowed_tools" in sdk_fields and options.allowed_tools:
            sdk_options["allowed_tools"] = list(options.allowed_tools)

        if "disallowed_tools" in sdk_fields and options.disallowed_tools:
            sdk_options["disallowed_tools"] = list(options.disallowed_tools)

        permission_mode = _normalize_permission_mode(options.permission_mode)
        if "permission_mode" in sdk_fields and permission_mode:
            sdk_options["permission_mode"] = permission_mode

        if "permission_prompt_tool_name" in sdk_fields and options.permission_prompt_tool_name:
            sdk_options["permission_prompt_tool_name"] = options.permission_prompt_tool_name

        if "cwd" in sdk_fields and options.cwd:
            sdk_options["cwd"] = options.cwd

        if (
            "continue_conversation" in sdk_fields
            and options.continue_conversation is not None
        ):
            sdk_options["continue_conversation"] = options.continue_conversation

        if "resume" in sdk_fields and options.resume:
            sdk_options["resume"] = options.resume

        if "include_partial_messages" in sdk_fields and options.include_partial_messages:
            sdk_options["include_partial_messages"] = True

        if "max_budget_usd" in sdk_fields and options.max_budget_usd is not None:
            sdk_options["max_budget_usd"] = options.max_budget_usd

        if "fallback_model" in sdk_fields and options.fallback_model:
            sdk_options["fallback_model"] = options.fallback_model

        add_dirs = _parse_optional_list(options.add_dirs)
        if "add_dirs" in sdk_fields and add_dirs:
            sdk_options["add_dirs"] = add_dirs

        setting_sources = _parse_optional_list(options.setting_sources)
        if "setting_sources" in sdk_fields and setting_sources:
            sdk_options["setting_sources"] = setting_sources

        betas = _parse_optional_list(options.betas)
        if "betas" in sdk_fields and betas:
            sdk_options["betas"] = betas

        if "settings" in sdk_fields and options.settings:
            sdk_options["settings"] = options.settings

        if "user" in sdk_fields and options.user_id:
            sdk_options["user"] = options.user_id
        elif options.user_id:
            ignored_options.append("user_id")

        if "effort" in sdk_fields and thinking_effort:
            sdk_options["effort"] = thinking_effort
        elif thinking_effort:
            ignored_options.append("thinking_effort")

        if "max_thinking_tokens" in sdk_fields and options.max_thinking_tokens is not None:
            sdk_options["max_thinking_tokens"] = options.max_thinking_tokens
        elif options.max_thinking_tokens is not None:
            ignored_options.append("max_thinking_tokens")

        if "thinking" in sdk_fields:
            if options.thinking is False:
                sdk_options["thinking"] = {"type": "disabled"}
            elif thinking_requested:
                budget = options.thinking_budget
                if budget is None and options.max_thinking_tokens is not None:
                    budget = options.max_thinking_tokens
                if self.supports_adaptive_thinking and budget is None:
                    sdk_options["thinking"] = {"type": "adaptive"}
                else:
                    sdk_options["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget or DEFAULT_THINKING_TOKENS,
                    }
        elif options.thinking is not None or options.thinking_budget is not None:
            ignored_options.append("thinking")

        schema_dict = _schema_to_dict(prompt.schema)
        schema_instruction = None

        output_format = _parse_optional_object(options.output_format)
        if schema_dict and output_format is None:
            output_format = {
                "type": "json_schema",
                "schema": schema_dict,
            }

        if output_format:
            if "output_format" in sdk_fields:
                sdk_options["output_format"] = output_format
            else:
                schema_instruction = (
                    "Please respond with valid JSON matching this schema:\n"
                    + json.dumps(output_format.get("schema", output_format))
                )

        if options.web_search:
            if not self.supports_web_search:
                raise ValueError(f"Web search is not supported by model {self.model_id}")
            if options.web_search_max_uses is not None:
                raise ValueError("web_search_max_uses is not supported by Claude SDK")
            if _parse_optional_list(options.web_search_allowed_domains):
                raise ValueError("web_search_allowed_domains is not supported by Claude SDK")
            if _parse_optional_list(options.web_search_blocked_domains):
                raise ValueError("web_search_blocked_domains is not supported by Claude SDK")
            if _parse_optional_object(options.web_search_location):
                raise ValueError("web_search_location is not supported by Claude SDK")

            if "allowed_tools" in sdk_fields:
                allowed = list(sdk_options.get("allowed_tools", []))
                if "WebSearch" not in allowed:
                    allowed.append("WebSearch")
                sdk_options["allowed_tools"] = allowed
            else:
                ignored_options.append("web_search")

        # Compatibility knobs currently not exposed by Claude SDK
        if options.max_tokens is not None:
            ignored_options.append("max_tokens")
        if options.temperature not in (None, DEFAULT_TEMPERATURE):
            ignored_options.append("temperature")
        if options.top_p is not None:
            ignored_options.append("top_p")
        if options.top_k is not None:
            ignored_options.append("top_k")
        if _parse_stop_sequences(options.stop_sequences):
            ignored_options.append("stop_sequences")
        if options.cache:
            ignored_options.append("cache")

        return SDKClaudeOptions(**sdk_options), ignored_options, schema_instruction

    def _extract_stream_event_text(self, message: StreamEvent, schema_expected: bool) -> str:
        event = getattr(message, "event", None)
        if not isinstance(event, dict):
            return ""
        if event.get("type") != "content_block_delta":
            return ""
        delta = event.get("delta", {})
        if not isinstance(delta, dict):
            return ""

        delta_type = delta.get("type")
        if delta_type == "text_delta":
            return delta.get("text") or ""
        if schema_expected and delta_type in ("input_json_delta", "json_delta"):
            return delta.get("partial_json") or delta.get("text") or ""
        return ""

    def _set_usage_from_sdk(
        self,
        response,
        usage: Optional[dict],
        prompt_text: str,
        output: str,
    ):
        if usage:
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")

            if not isinstance(input_tokens, int):
                alt_input = usage.get("inputTokens")
                input_tokens = alt_input if isinstance(alt_input, int) else None
            if not isinstance(output_tokens, int):
                alt_output = usage.get("outputTokens")
                output_tokens = alt_output if isinstance(alt_output, int) else None

            details = {
                key: value
                for key, value in usage.items()
                if key not in ("input_tokens", "output_tokens", "inputTokens", "outputTokens")
            }
            response.set_usage(
                input=input_tokens,
                output=output_tokens,
                details=details or None,
            )
            return

        response.set_usage(
            input=int(len(prompt_text.split()) * 1.3),
            output=int(len(output.split()) * 1.3),
        )

    def _serialize_message(self, message: Any) -> Dict[str, Any]:
        if isinstance(message, AssistantMessage):
            content = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingBlock):
                    content.append({"type": "thinking", "thinking": block.thinking})
                elif isinstance(block, ToolUseBlock):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )
            return {
                "type": "assistant",
                "model": getattr(message, "model", None),
                "content": content,
            }

        if isinstance(message, UserMessage):
            return {
                "type": "user",
                "content": getattr(message, "content", None),
            }

        if isinstance(message, SystemMessage):
            return {
                "type": "system",
                "subtype": getattr(message, "subtype", None),
            }

        if isinstance(message, ResultMessage):
            return {
                "type": "result",
                "subtype": message.subtype,
                "duration_ms": message.duration_ms,
                "duration_api_ms": message.duration_api_ms,
                "is_error": message.is_error,
                "num_turns": message.num_turns,
                "session_id": message.session_id,
                "total_cost_usd": message.total_cost_usd,
                "usage": message.usage,
                "result": message.result,
                "structured_output": getattr(message, "structured_output", None),
            }

        if isinstance(message, StreamEvent):
            return {
                "type": "stream_event",
                "session_id": message.session_id,
                "uuid": message.uuid,
                "event": message.event,
            }

        return {"type": type(message).__name__}

    async def _execute_async_impl(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response,
        conversation,
        key: Optional[str],
    ) -> AsyncIterator[str]:
        if prompt.schema and prompt.tools:
            raise ValueError("Cannot use both schema and tools in the same prompt")

        api_key = self.get_key(key)
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or run 'llm keys set anthropic'."
            )
        os.environ[self.key_env_var] = api_key

        options = self._prompt_options(prompt)
        messages = self._build_messages(prompt, conversation)
        sdk_options, ignored_options, schema_instruction = self._build_options(prompt)
        query_prompt, prompt_mode = self._build_query_prompt(messages, schema_instruction)

        response._prompt_json = {
            "mode": prompt_mode,
            "messages": messages,
        }

        chunks: List[str] = []
        response_messages: List[Dict[str, Any]] = []
        usage_data: Optional[dict] = None
        saw_text = False
        result_structured_output: Any = None
        prefill = self._prefill_text(prompt)

        if prefill:
            chunks.append(prefill)
            if stream:
                yield prefill

        async for message in query(prompt=query_prompt, options=sdk_options):
            response_messages.append(self._serialize_message(message))

            if isinstance(message, AssistantMessage):
                if getattr(message, "model", None):
                    response.resolved_model = message.model
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        saw_text = True
                        chunks.append(block.text)
                        if stream:
                            yield block.text
                    elif isinstance(block, ToolUseBlock):
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=block.id,
                                name=block.name,
                                arguments=block.input or {},
                            )
                        )

            elif isinstance(message, StreamEvent):
                partial_text = self._extract_stream_event_text(
                    message,
                    schema_expected=bool(prompt.schema),
                )
                if partial_text and options.include_partial_messages:
                    saw_text = True
                    chunks.append(partial_text)
                    if stream:
                        yield partial_text

            elif isinstance(message, ResultMessage):
                usage_data = message.usage
                result_structured_output = getattr(message, "structured_output", None)
                if message.is_error:
                    raise RuntimeError(message.result or "Claude SDK returned an error")

                if result_structured_output is not None and prompt.schema and not saw_text:
                    serialized = json.dumps(result_structured_output)
                    saw_text = True
                    chunks.append(serialized)
                    if stream:
                        yield serialized

                if message.result and not saw_text:
                    saw_text = True
                    chunks.append(message.result)
                    if stream:
                        yield message.result

        final_text = "".join(chunks)
        if not stream and final_text:
            yield final_text

        self._set_usage_from_sdk(response, usage_data, str(response._prompt_json), final_text)
        response.response_json = {
            "model": self.claude_model_id,
            "sdk": SDK_NAME,
            "messages": response_messages,
            "ignored_options": sorted(set(ignored_options)),
            "prompt_mode": prompt_mode,
            "structured_output": result_structured_output,
        }


class ClaudeCodeMessages(_Shared, llm.KeyModel):
    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation] = None,
        key: Optional[str] = None,
    ) -> Iterator[str]:
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def run_async():
            async def _consume():
                try:
                    async for chunk in self._execute_async_impl(
                        prompt,
                        stream,
                        response,
                        conversation,
                        key,
                    ):
                        events.put(("chunk", chunk))
                except Exception as ex:
                    events.put(("error", ex))
                finally:
                    events.put(("done", None))

            asyncio.run(_consume())

        worker = threading.Thread(target=run_async, daemon=True)
        worker.start()

        while True:
            event_type, payload = events.get()
            if event_type == "chunk":
                yield payload
            elif event_type == "error":
                worker.join(timeout=0.1)
                raise payload
            else:
                break
        worker.join()


class AsyncClaudeCodeMessages(_Shared, llm.AsyncKeyModel):
    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: Optional[llm.AsyncConversation] = None,
        key: Optional[str] = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._execute_async_impl(
            prompt,
            stream,
            response,
            conversation,
            key,
        ):
            yield chunk


@llm.hookimpl
def register_models(register):
    for model_id, config in MODEL_CONFIGS.items():
        aliases = config.get("aliases", [])
        model_config = {k: v for k, v in config.items() if k != "aliases"}
        register(
            ClaudeCodeMessages(model_id, model_config),
            AsyncClaudeCodeMessages(model_id, model_config),
            aliases=tuple(aliases) if aliases else None,
        )
