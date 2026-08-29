#!/usr/bin/env bash
# Start the Codex CLI -> Kimi-K3 (chat/completions) translation proxy.
#
# Required:
#   export KIMI_API_KEY=...   # Meta/Llama API bearer token (K_LLAMA)
#
# Optional:
#   CODEX_KIMI_PROXY_HOST  (default 127.0.0.1)
#   CODEX_KIMI_PROXY_PORT  (default 3458)
#   CODEX_KIMI_UPSTREAM    (default https://api.llama.com/experimental/compat/openai/v1)
#   CODEX_KIMI_DEBUG=0     (disable /tmp debug log)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

: "${KIMI_API_KEY:?set KIMI_API_KEY to your Meta/Llama API key}"

export CODEX_KIMI_PROXY_HOST="${CODEX_KIMI_PROXY_HOST:-127.0.0.1}"
export CODEX_KIMI_PROXY_PORT="${CODEX_KIMI_PROXY_PORT:-3458}"
export CODEX_KIMI_UPSTREAM="${CODEX_KIMI_UPSTREAM:-https://api.llama.com/experimental/compat/openai/v1}"

if curl -fsS "http://${CODEX_KIMI_PROXY_HOST}:${CODEX_KIMI_PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "proxy already running on ${CODEX_KIMI_PROXY_HOST}:${CODEX_KIMI_PROXY_PORT}"
  exit 0
fi

exec python3 "$ROOT/codex_kimi_proxy.py"
