# llm-claude-code

[![Changelog](https://img.shields.io/github/v/release/VanL/llm-claude-code?include_prereleases&label=changelog)](https://github.com/VanL/llm-claude-code/releases)
[![Tests](https://github.com/VanL/llm-claude-code/actions/workflows/test.yml/badge.svg)](https://github.com/VanL/llm-claude-code/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/VanL/llm-claude-code/blob/main/LICENSE)

LLM plugin for accessing Claude models through the [claude-code-sdk](https://github.com/anthropics/claude-code-sdk-python)

## Overview

This plugin provides access to Claude models via the Claude Code SDK instead of the standard Anthropic API. It routes all requests through the SDK, which enables:

- Integration with Claude Code's development-focused features
- Access to Claude 4 Opus and Sonnet models
- Support for system prompts and conversation history
- Tool/function calling through the SDK's allowed_tools mechanism

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/) from source:
```bash
git clone https://github.com/VanL/llm-claude-code
cd llm-claude-code
pip install -e .
```

## Prerequisites

- Python 3.10+
- Node.js (required by claude-code-sdk)
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

This plugin provides access to the following models through the Claude Code SDK:

- `claude-code-opus` (or `claude-code-opus-4-20250514`) - Claude 4 Opus
- `claude-code-sonnet` (or `claude-code-sonnet-4-20250514`) - Claude 4 Sonnet

Run prompts like this:
```bash
llm -m claude-code-opus 'Fun facts about walruses'
llm -m claude-code-sonnet 'Fun facts about pelicans'
```

### Working with Code

Since this uses the Claude Code SDK, it's particularly well-suited for coding tasks:
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

- **max_tokens**: `int` - Maximum number of tokens to generate
- **temperature**: `float` - Sampling temperature (0.0-1.0)
- **top_p**: `float` - Top-p sampling parameter
- **top_k**: `int` - Top-k sampling parameter
- **stop_sequences**: `str` or `list` - Sequences that stop generation
- **system_prompt**: `str` - System prompt to use
- **cwd**: `str` - Working directory for Claude Code

Note: Some features from the standard Anthropic API (like prefill, cache, user_id) are tracked but may not be fully functional through the SDK.

## Differences from llm-anthropic

This plugin differs from the standard `llm-anthropic` plugin in several ways:

1. **Model Access**: Only Opus and Sonnet models are available through the SDK
2. **Model Names**: All models are prefixed with `claude-code-` to avoid conflicts
3. **SDK Routing**: All requests go through the claude-code-sdk instead of direct API calls
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

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Claude Code SDK responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_ANTHROPIC_API_KEY="$(llm keys get anthropic)" pytest --record-mode once
```

## Troubleshooting

### "No claude-code-sdk found" error
Make sure you have installed the claude-code-sdk:
```bash
pip install claude-code-sdk
```

### "No Anthropic API key found" error
Set your API key using:
```bash
llm keys set anthropic
```
Or set the `ANTHROPIC_API_KEY` environment variable.

### Node.js errors
The claude-code-sdk requires Node.js. Make sure you have it installed:
```bash
node --version
```

## License

Apache License 2.0