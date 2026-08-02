#!/usr/bin/env bash
# Start 1rgs/claude-code-proxy pointed at Meta Llama openai-compat (Opus).
set -euo pipefail
PROXY_ROOT="${PROXY_ROOT:-/home/songyang/projects/aginfer/claude-code-proxy}"
HOST="${CCP_HOST:-127.0.0.1}"
PORT="${CCP_PORT:-8082}"

if curl -fsS -m 2 "http://${HOST}:${PORT}/docs" >/dev/null 2>&1 \
  || [[ "$(curl -sS -m 2 -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/" || true)" != "000" ]]; then
  echo "claude-code-proxy already up on ${HOST}:${PORT}"
  exit 0
fi

cd "$PROXY_ROOT"
if [[ ! -f .env ]]; then
  echo "missing $PROXY_ROOT/.env" >&2
  exit 1
fi

echo "starting claude-code-proxy on ${HOST}:${PORT} (Meta Opus via OPENAI_BASE_URL)"
# no --reload: keep mapping patch stable under load
exec uv run uvicorn server:app --host "$HOST" --port "$PORT" --log-level warning \
  >>/tmp/claude_code_proxy_meta.log 2>&1
