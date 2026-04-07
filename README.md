# llm-claude-code

[![Changelog](https://img.shields.io/github/v/release/VanL/llm-claude-code?include_prereleases&label=changelog)](https://github.com/VanL/llm-claude-code/releases)
[![Tests](https://github.com/VanL/llm-claude-code/actions/workflows/test.yml/badge.svg)](https://github.com/VanL/llm-claude-code/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/VanL/llm-claude-code/blob/main/LICENSE)

LLM plugin for accessing Claude models through the [Claude Agent SDK](https://docs.claude.com/en/docs/claude-code/sdk)

## Overview

This plugin provides access to Claude models via the Claude Agent SDK instead of the standard Anthropic API. It routes all requests through the SDK, which enables:

- Integration with Claude Code's development-focused features
- Access to current Claude Opus, Sonnet, and Haiku model families
- Support for system prompts and conversation history
- Tool/function calling through the SDK's allowed_tools mechanism

Parity progress against `llm-anthropic` is tracked in [PARITY_MATRIX.md](PARITY_MATRIX.md).

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/) from source:
```bash
git clone https://github.com/VanL/llm-claude-code
cd llm-claude-code
pip install -e .
```

## Prerequisites

- Python 3.10+
- Node.js (required by Claude Code / Agent SDK)
- Claude Code CLI installed globally

## Usage

First, set [an API key](https://console.anthropic.com/settings/keys) for Anthropic:
```bash
llm keys set anthropic
# Paste key here
```

You can also set the key in the environment variable `ANTHROPIC_API_KEY`

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

### Available Models

This plugin provides access to the following models through the Claude SDK:

- `claude-code-opus` - latest Opus model exposed by Claude Code (currently `claude-code-opus-4-6`)
- `claude-code-sonnet` - latest Sonnet model exposed by Claude Code (currently `claude-code-sonnet-4-6`)
- `claude-code-haiku` - latest Haiku model exposed by Claude Code (currently `claude-code-haiku-4-5-20251001`)
- Explicit versioned models for `4.0`, `4.1`, `4.5`, and `4.6` where Claude Code exposes them

Run prompts like this:
```bash
llm -m claude-code-opus 'Fun facts about walruses'
llm -m claude-code-sonnet 'Fun facts about pelicans'
llm -m claude-code-haiku 'Fun facts about cormorants'
llm -m claude-code-opus-4.5 'Fun facts about capybaras'
```

### Working with Code

Since this uses the Claude SDK, it's particularly well-suited for coding tasks:
```bash
llm -m claude-code-sonnet 'Write a Python function to calculate fibonacci numbers'
```

### System Prompts

You can set system prompts to guide the model's behavior:
```bash
llm -m claude-code-opus 'Review this code' --system 'You are an expert code reviewer'
```

### Conversations

The plugin supports multi-turn conversations:
```bash
llm -m claude-code-sonnet --continue
# This starts an interactive conversation
```

## Model Options

The following options can be passed using `-o name value` on the CLI:

- **system_prompt**: `str` - System prompt to use
- **append_system_prompt**: `str` - Append additional system guidance
- **max_turns**: `int` - Maximum SDK conversation turns
- **allowed_tools**: `list[str]` - Allowed Claude Code tools
- **disallowed_tools**: `list[str]` - Disallowed Claude Code tools
- **permission_mode**: `str` - Tool permission mode (`default`, `acceptEdits`, `plan`, `bypassPermissions`)
- **cwd**: `str` - Working directory for Claude Code
- **add_dirs**: `list[str]` - Additional directories Claude can access
- **resume**: `str` - Resume a previous Claude session
- **continue_conversation**: `bool` - Continue current conversation state
- **include_partial_messages**: `bool` - Stream partial message events
- **max_budget_usd**: `float` - Optional spend guard for a run
- **thinking**: `bool` - Enable extended thinking; 4.6 models use adaptive thinking by default
- **thinking_budget**: `int` - Anthropic-compatible thinking budget; also enables thinking when set
- **max_thinking_tokens**: `int` - SDK-native direct thinking token limit
- **thinking_effort**: `str` - Compatibility alias for SDK `effort`
- **web_search**: `bool` - Enables Claude Code web search tool shim (`WebSearch`)
- **user_id**: `str` - Mapped to SDK `user` when supported

Compatibility caveats:
- Anthropic API sampling/cache knobs (`max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `cache`) do not have direct Claude SDK equivalents and are reported in `response_json["ignored_options"]`.
- Advanced web-search constraints (`web_search_allowed_domains`, `web_search_blocked_domains`, `web_search_location`, `web_search_max_uses`) are currently rejected.
- Claude 4.6 assistant-prefill prompts are rejected up front to match provider behavior.

## Differences from llm-anthropic

This plugin differs from the standard `llm-anthropic` plugin in several ways:

1. **Model Access**: Only Claude Code-supported models are available through the SDK
2. **Model Names**: All models are prefixed with `claude-code-` to avoid conflicts
3. **SDK Routing**: All requests go through the Claude SDK instead of direct API calls
4. **Feature Support**: Some features like image attachments, PDFs, and structured output may work differently or have limited support
5. **Tool Calling**: Uses the SDK's `allowed_tools` mechanism instead of standard function calling

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-claude-code
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
pytest
```

The tests are unit tests that mock SDK responses and do not require live API calls.

## Troubleshooting

### "No Claude SDK found" error
Make sure you have installed the Claude Agent SDK:
```bash
pip install claude-agent-sdk
```

### "No Anthropic API key found" error
Set your API key using:
```bash
llm keys set anthropic
```
Or set the `ANTHROPIC_API_KEY` environment variable.

### Node.js errors
The Claude SDK requires Node.js. Make sure you have it installed:
```bash
node --version
```

## License

Apache License 2.0
