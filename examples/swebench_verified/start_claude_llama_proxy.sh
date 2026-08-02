#!/usr/bin/env bash
# Start Anthropic→Llama OpenAI-compat proxy for Claude Code.
#
# Required:
#   export CLAUDE_LLAMA_API_KEY=...   # Meta / Llama API bearer token
#
# Optional:
#   CLAUDE_LLAMA_PROXY_HOST  (default 127.0.0.1)
#   CLAUDE_LLAMA_PROXY_PORT  (default 3456)
#   CLAUDE_LLAMA_UPSTREAM    (default Meta openai-compat base)
#   CLAUDE_LLAMA_MODEL       (default claude-4-8-opus-genai)
#   CLAUDE_LLAMA_FORCE_MODEL (override all client model ids)
#   CLAUDE_LLAMA_DEBUG=0     (disable /tmp debug log)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

: "${CLAUDE_LLAMA_API_KEY:?set CLAUDE_LLAMA_API_KEY to your Meta/Llama API key}"

export CLAUDE_LLAMA_PROXY_HOST="${CLAUDE_LLAMA_PROXY_HOST:-127.0.0.1}"
export CLAUDE_LLAMA_PROXY_PORT="${CLAUDE_LLAMA_PROXY_PORT:-3456}"
export CLAUDE_LLAMA_UPSTREAM="${CLAUDE_LLAMA_UPSTREAM:-https://api.llama.com/experimental/compat/openai/v1}"
export CLAUDE_LLAMA_MODEL="${CLAUDE_LLAMA_MODEL:-claude-4-8-opus-genai}"

if curl -fsS "http://${CLAUDE_LLAMA_PROXY_HOST}:${CLAUDE_LLAMA_PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "proxy already running on ${CLAUDE_LLAMA_PROXY_HOST}:${CLAUDE_LLAMA_PROXY_PORT}"
  exit 0
fi

exec python3 "$ROOT/claude_llama_proxy.py"
