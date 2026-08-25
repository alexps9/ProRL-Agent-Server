#!/usr/bin/env bash
# Start the Codex CLI -> Meta Model API proxy.
#
# Required:
#   export CODEX_META_API_KEY=...   # Meta Model API bearer token (K_LLAMA)
#
# Optional:
#   CODEX_META_PROXY_HOST  (default 127.0.0.1)
#   CODEX_META_PROXY_PORT  (default 3457)
#   CODEX_META_UPSTREAM    (default https://api.ai.meta.com/v1)
#   CODEX_META_DEBUG=0     (disable /tmp debug log)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

: "${CODEX_META_API_KEY:?set CODEX_META_API_KEY to your Meta Model API key}"

export CODEX_META_PROXY_HOST="${CODEX_META_PROXY_HOST:-127.0.0.1}"
export CODEX_META_PROXY_PORT="${CODEX_META_PROXY_PORT:-3457}"
export CODEX_META_UPSTREAM="${CODEX_META_UPSTREAM:-https://api.ai.meta.com/v1}"

if curl -fsS "http://${CODEX_META_PROXY_HOST}:${CODEX_META_PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "proxy already running on ${CODEX_META_PROXY_HOST}:${CODEX_META_PROXY_PORT}"
  exit 0
fi

exec python3 "$ROOT/codex_meta_proxy.py"
