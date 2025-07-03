"""
LLM plugin for Claude Code SDK - provides access to Claude models through the claude-code-sdk
"""
import asyncio
import base64
import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Set, Union

import llm
from llm.models import model_from_llm_id
from llm.types import Options
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from claude_code_sdk import (
        query,
        ClaudeCodeOptions as SDKClaudeCodeOptions,
        AssistantMessage,
        TextBlock,
        ToolUseBlock,
        ToolResultBlock,
        UserMessage,
        SystemMessage,
        ResultMessage,
    )
except ImportError:
    raise ImportError(
        "You need to install the claude-code-sdk package: pip install claude-code-sdk"
    )

DEFAULT_THINKING_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0


# Model configurations with their capabilities
# Only models available through Claude Code SDK
MODEL_CONFIGS = {
    # Current Claude 4 models (available through SDK)
    "claude-code-opus-4-20250514": {
        "claude_model_id": "claude-opus-4-20250514",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "default_max_tokens": 8192,
        "aliases": ["claude-code-opus", "claude-code-opus-4"],
    },
    "claude-code-sonnet-4-20250514": {
        "claude_model_id": "claude-sonnet-4-20250514",
        "supports_images": True,
        "supports_pdf": True,
        "supports_thinking": True,
        "default_max_tokens": 8192,
        "aliases": ["claude-code-sonnet", "claude-code-sonnet-4"],
    },
}

def source_for_attachment(attachment):
    """Convert attachment to format suitable for Claude Code SDK"""
    if attachment.url:
        return {
            "type": "url",
            "url": attachment.url,
        }
    else:
        return {
            "data": attachment.base64_content(),
            "media_type": attachment.resolve_type(),
            "type": "base64",
        }


class ClaudeCodeOptions(BaseModel):
    """Options for Claude Code models"""

    # Claude Code SDK supported options
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to use",
    )
    max_turns: Optional[int] = Field(
        default=1,
        description="Maximum number of conversation turns",
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="List of allowed tools for Claude Code to use",
    )
    permission_mode: Optional[str] = Field(
        default=None,
        description="Permission mode for tool usage ('ask', 'acceptEdits', 'acceptAll')",
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory for Claude Code",
    )
    
    # Standard Anthropic API options (not directly supported by SDK, but we'll track them)
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter",
    )
    stop_sequences: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Sequences that stop generation",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="A UUID representing your end-user (for Anthropic's safety monitoring)",
    )
    prefill: Optional[str] = Field(
        default=None,
        description="Start of Claude's response to prefill",
    )
    hide_prefill: Optional[bool] = Field(
        default=False,
        description="Hide prefill from output",
    )
    cache: Optional[bool] = Field(
        default=None,
        description="Enable prompt caching",
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if temperature == 0:
            return 0
        return temperature or DEFAULT_TEMPERATURE

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, stop_sequences):
        if stop_sequences is None:
            return None
        if isinstance(stop_sequences, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(stop_sequences)
                if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            # Otherwise treat as single stop sequence
            return [stop_sequences]
        elif isinstance(stop_sequences, list):
            if all(isinstance(s, str) for s in stop_sequences):
                return stop_sequences
            else:
                raise ValueError("stop_sequences must be a list of strings")
        else:
            raise ValueError(
                "stop_sequences must be a string, a JSON array of strings, or a list"
            )
        return stop_sequences


class ClaudeCodeOptionsWithThinking(ClaudeCodeOptions):
    """Options for Claude Code models with thinking support"""

    thinking: Optional[bool] = Field(
        default=None,
        description="Enable thinking mode",
    )
    thinking_budget: Optional[int] = Field(
        default=1024,
        description="Token budget for thinking",
    )


class _Shared:
    """Shared functionality between sync and async implementations"""

    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True
    
    # Default capabilities
    supports_schema = True  # SDK may not directly support, but we can work with structured prompts
    supports_tools = True   # SDK supports tools through allowed_tools

    def __init__(self, model_id: str, config: dict):
        self.model_id = "anthropic/" + model_id  # Prefix for compatibility
        # Use the claude_model_id from config if provided, otherwise extract from model_id
        self.claude_model_id = config.get("claude_model_id", model_id.replace("claude-code-", "claude-"))
        self.supports_images = config["supports_images"]
        self.supports_pdf = config["supports_pdf"]
        self.supports_thinking = config.get("supports_thinking", False)
        self.default_max_tokens = config.get("default_max_tokens", 4096)
        self._api_key = None
        
        # Build attachment types set
        self.attachment_types = set()
        if self.supports_images:
            self.attachment_types.update({
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
            })
        if self.supports_pdf:
            self.attachment_types.add("application/pdf")
            
        # Set options class based on capabilities
        if self.supports_thinking:
            self.Options = ClaudeCodeOptionsWithThinking
        else:
            self.Options = ClaudeCodeOptions

    def get_key(self, key=None):
        """Get API key from parameter, env var, or key store"""
        if key:
            return key

        # Check environment variable
        from_env = os.environ.get(self.key_env_var)
        if from_env:
            return from_env

        # Check key store - llm framework passes this to us
        return self._api_key
    
    def prefill_text(self, prompt):
        """Get prefill text if specified and not hidden"""
        if hasattr(prompt.options, "prefill") and prompt.options.prefill:
            if not (hasattr(prompt.options, "hide_prefill") and prompt.options.hide_prefill):
                return prompt.options.prefill
        return ""

    def build_options(self, prompt: llm.Prompt, stream: bool) -> SDKClaudeCodeOptions:
        """Build SDK options from prompt options"""
        options = prompt.options or self.Options()
        
        # Extract system prompt from conversation if present
        system_prompt = None
        if prompt.conversation:
            for response in prompt.conversation.responses:
                if response.prompt.system:
                    system_prompt = response.prompt.system
                    break
        
        # Override with explicit system option if provided
        if prompt.system:
            system_prompt = prompt.system

        # Build the SDK options with only supported fields
        sdk_options = {
            "system_prompt": system_prompt,
            "max_turns": 1,  # LLM interface expects single turn responses
        }
        
        # Add working directory if specified
        if hasattr(options, "cwd") and options.cwd:
            sdk_options["cwd"] = options.cwd

        # Handle tool/function calling through allowed_tools
        if prompt.tools or (prompt.conversation and hasattr(prompt.conversation, "tools") and prompt.conversation.tools):
            # Map LLM tools to Claude Code allowed tools
            # This is a simplified mapping - you may need to adjust based on tool names
            sdk_options["allowed_tools"] = ["Bash", "Read", "Write", "Edit"]
            sdk_options["permission_mode"] = "ask"
        elif hasattr(options, "allowed_tools") and options.allowed_tools:
            sdk_options["allowed_tools"] = options.allowed_tools
            if hasattr(options, "permission_mode") and options.permission_mode:
                sdk_options["permission_mode"] = options.permission_mode

        return SDKClaudeCodeOptions(**sdk_options)

    def build_messages(self, prompt: llm.Prompt, conversation=None) -> List[dict]:
        """Build messages array for conversation history - not used by SDK but kept for compatibility"""
        messages = []
        # The Claude Code SDK handles conversation differently, so we'll just track this for logging
        return messages
    
    def build_prompt_with_conversation(self, prompt: llm.Prompt) -> str:
        """Build a complete prompt including conversation history and attachments"""
        parts = []
        
        # Add conversation history if present
        if prompt.conversation:
            for response in prompt.conversation.responses:
                # Add user message
                if response.prompt.prompt:
                    parts.append(f"Human: {response.prompt.prompt}")
                
                # Add assistant response
                if response.text():
                    parts.append(f"Assistant: {response.text()}")
                    
                # Add tool results if any
                if hasattr(response.prompt, "tool_results") and response.prompt.tool_results:
                    for tool_result in response.prompt.tool_results:
                        parts.append(f"Tool Result ({tool_result.tool_call_id}): {tool_result.output}")
        
        # Add current prompt with prefill support
        current_parts = []
        if prompt.prompt:
            current_parts.append(prompt.prompt)
        
        # Handle attachments (SDK doesn't directly support these, so we describe them)
        if prompt.attachments:
            for attachment in prompt.attachments:
                attachment_type = attachment.resolve_type()
                
                if attachment_type in self.attachment_types:
                    if attachment_type.startswith("image/") and self.supports_images:
                        # For images, we'll need to handle them differently with Claude Code
                        current_parts.append(f"[Image: {attachment.path or 'embedded image'}]")
                    elif attachment_type == "application/pdf" and self.supports_pdf:
                        current_parts.append(f"[PDF Document: {attachment.path or 'embedded PDF'}]")
                elif attachment_type == "text/plain" or attachment_type is None:
                    # For text files, we can include the content directly
                    try:
                        content = attachment.read_text()
                        current_parts.append(f"[File content from {attachment.path or 'attachment'}]:\n{content}")
                    except:
                        current_parts.append(f"[Text file: {attachment.path or 'attachment'}]")
        
        # Add prefill if specified
        prefill = self.prefill_text(prompt)
        if prefill:
            current_parts.append(f"\nAssistant: {prefill}")
        
        if current_parts:
            parts.append("Human: " + "\n".join(current_parts))
        
        return "\n\n".join(parts)

    def process_response_content(self, message, response=None) -> tuple[str, dict]:
        """Process a message from Claude Code SDK and extract text content and metadata"""
        text_parts = []
        metadata = {"message_type": type(message).__name__}
        
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    # Record tool usage and add to response if available
                    tool_call_data = {
                        "name": getattr(block, "name", "unknown"),
                        "id": getattr(block, "id", None),
                        "arguments": getattr(block, "input", {}),
                    }
                    
                    if "tool_uses" not in metadata:
                        metadata["tool_uses"] = []
                    metadata["tool_uses"].append(tool_call_data)
                    
                    # Add tool call to response object if available
                    if response and hasattr(response, "add_tool_call"):
                        response.add_tool_call(
                            id=tool_call_data["id"],
                            name=tool_call_data["name"],
                            arguments=tool_call_data["arguments"],
                        )
        elif isinstance(message, UserMessage):
            # Handle user messages if they come through
            if hasattr(message, "content") and isinstance(message.content, str):
                text_parts.append(message.content)
        elif isinstance(message, SystemMessage):
            # System messages typically don't need to be shown
            metadata["system_message"] = True
        elif isinstance(message, ResultMessage):
            # Tool results
            if hasattr(message, "content"):
                text_parts.append(str(message.content))
        
        return "\n".join(text_parts), metadata


class ClaudeCodeMessages(_Shared, llm.KeyModel):
    """Synchronous Claude Code model implementation"""

    can_stream = True

    def __init__(self, model_id: str, config: dict):
        super().__init__(model_id, config)

    def get_options_class(self) -> type:
        """Return the appropriate options class based on model capabilities"""
        if self.supports_thinking:
            return ClaudeCodeOptionsWithThinking
        return ClaudeCodeOptions

    def prompt_blocks(self):
        yield llm.remove_role_block()
        yield llm.multi_binary_attachments_block()

    def stream(self, prompt: llm.Prompt, stream: bool, **kwargs) -> Iterator[str]:
        """Stream responses from Claude Code SDK"""
        api_key = self.get_key(kwargs.get("key"))
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or use 'llm keys set anthropic'"
            )

        # Set the API key for the SDK
        os.environ["ANTHROPIC_API_KEY"] = api_key

        # Build options
        options = self.build_options(prompt, stream)
        
        # Build prompt with conversation history
        full_prompt = self.build_prompt_with_conversation(prompt)
        
        # Check for schema + tools conflict
        if prompt.schema and prompt.tools:
            raise ValueError("Cannot use both schema and tools in the same prompt")
        
        # Handle structured output if schema is provided
        if prompt.schema:
            # Append schema instruction to prompt
            schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(prompt.schema)}\n"
            full_prompt += schema_instruction

        # Run the async query in a sync context
        async def _async_stream():
            accumulated_text = []
            full_response_json = {}
            usage_data = {}
            prefill_stripped = False
            
            # Add prefill to accumulated text if needed
            prefill = self.prefill_text(prompt)
            if prefill:
                if not prompt.options.hide_prefill:
                    accumulated_text.append(prefill)
                    if stream:
                        yield prefill
                else:
                    prefill_stripped = True
            
            async for message in query(prompt=full_prompt, options=options):
                text, metadata = self.process_response_content(message, prompt._response if hasattr(prompt, "_response") else None)
                
                # Skip empty text from prefill stripping
                if prefill_stripped and not text:
                    prefill_stripped = False
                    continue
                    
                if text:
                    accumulated_text.append(text)
                    if stream:
                        yield text
                
                # Build response JSON structure
                if isinstance(message, AssistantMessage):
                    # Try to extract model info and usage if available
                    full_response_json["model"] = self.claude_model_id
                    full_response_json["message_type"] = "claude_code_response"
                    
                    # Store content blocks
                    if "content" not in full_response_json:
                        full_response_json["content"] = []
                    
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            full_response_json["content"].append({
                                "type": "text",
                                "text": block.text
                            })
                        elif isinstance(block, ToolUseBlock):
                            full_response_json["content"].append({
                                "type": "tool_use",
                                "id": getattr(block, "id", None),
                                "name": getattr(block, "name", "unknown"),
                                "input": getattr(block, "input", {})
                            })
            
            # If not streaming, yield all accumulated text at once
            if not stream and accumulated_text:
                # Strip prefill if hidden
                final_text = "".join(accumulated_text)
                if prompt.options.hide_prefill and prefill:
                    final_text = final_text[len(prefill):]
                yield final_text
            
            # Set usage data (estimate since SDK doesn't provide it)
            if hasattr(prompt, "_response") and prompt._response:
                # Estimate token counts (rough approximation)
                input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
                output_tokens = len("".join(accumulated_text).split()) * 1.3
                
                prompt._response.set_usage(
                    input=int(input_tokens),
                    output=int(output_tokens)
                )
                
                # Store response JSON
                prompt._response.response_json = full_response_json

        # Create event loop if needed and run the async generator
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Convert async generator to sync
        async_gen = _async_stream()
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    def stream_method(self) -> str:
        return "stream"


class AsyncClaudeCodeMessages(_Shared, llm.AsyncKeyModel):
    """Asynchronous Claude Code model implementation"""

    def __init__(self, model_id: str, config: dict):
        super().__init__(model_id, config)

    def get_options_class(self) -> type:
        """Return the appropriate options class based on model capabilities"""
        if self.supports_thinking:
            return ClaudeCodeOptionsWithThinking
        return ClaudeCodeOptions

    def prompt_blocks(self):
        yield llm.remove_role_block()
        yield llm.multi_binary_attachments_block()

    async def stream(
        self, prompt: llm.Prompt, stream: bool, **kwargs
    ) -> AsyncIterator[str]:
        """Stream responses from Claude Code SDK asynchronously"""
        api_key = self.get_key(kwargs.get("key"))
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or use 'llm keys set anthropic'"
            )

        # Set the API key for the SDK
        os.environ["ANTHROPIC_API_KEY"] = api_key

        # Build options
        options = self.build_options(prompt, stream)
        
        # Build prompt with conversation history
        full_prompt = self.build_prompt_with_conversation(prompt)
        
        # Check for schema + tools conflict
        if prompt.schema and prompt.tools:
            raise ValueError("Cannot use both schema and tools in the same prompt")
        
        # Handle structured output if schema is provided
        if prompt.schema:
            # Append schema instruction to prompt
            schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(prompt.schema)}\n"
            full_prompt += schema_instruction

        accumulated_text = []
        full_response_json = {}
        usage_data = {}
        prefill_stripped = False
        
        # Add prefill to accumulated text if needed
        prefill = self.prefill_text(prompt)
        if prefill:
            if not prompt.options.hide_prefill:
                accumulated_text.append(prefill)
                if stream:
                    yield prefill
            else:
                prefill_stripped = True
        
        async for message in query(prompt=full_prompt, options=options):
            text, metadata = self.process_response_content(message, prompt._response if hasattr(prompt, "_response") else None)
            
            # Skip empty text from prefill stripping
            if prefill_stripped and not text:
                prefill_stripped = False
                continue
                
            if text:
                accumulated_text.append(text)
                if stream:
                    yield text
            
            # Build response JSON structure
            if isinstance(message, AssistantMessage):
                # Try to extract model info and usage if available
                full_response_json["model"] = self.claude_model_id
                full_response_json["message_type"] = "claude_code_response"
                
                # Store content blocks
                if "content" not in full_response_json:
                    full_response_json["content"] = []
                
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_response_json["content"].append({
                            "type": "text",
                            "text": block.text
                        })
                    elif isinstance(block, ToolUseBlock):
                        full_response_json["content"].append({
                            "type": "tool_use",
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", "unknown"),
                            "input": getattr(block, "input", {})
                        })
        
        # If not streaming, yield all accumulated text at once
        if not stream and accumulated_text:
            # Strip prefill if hidden
            final_text = "".join(accumulated_text)
            if prompt.options.hide_prefill and prefill:
                final_text = final_text[len(prefill):]
            yield final_text
        
        # Set usage data (estimate since SDK doesn't provide it)
        if hasattr(prompt, "_response") and prompt._response:
            # Estimate token counts (rough approximation)
            input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
            output_tokens = len("".join(accumulated_text).split()) * 1.3
            
            prompt._response.set_usage(
                input=int(input_tokens),
                output=int(output_tokens)
            )
            
            # Store response JSON
            prompt._response.response_json = full_response_json

    def stream_method(self) -> str:
        return "stream"


@llm.hookimpl
def register_models(register):
    """Register all Claude Code models with the LLM plugin system"""
    for model_id, config in MODEL_CONFIGS.items():
        # Extract aliases from config
        aliases = config.get("aliases", [])
        
        # Create a copy of config without aliases for passing to model
        model_config = {k: v for k, v in config.items() if k != "aliases"}
        
        # Register sync models
        register(
            ClaudeCodeMessages(model_id, model_config),
            AsyncClaudeCodeMessages(model_id, model_config),
            aliases=tuple(aliases) if aliases else None,
        )