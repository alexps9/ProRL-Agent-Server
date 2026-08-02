#!/usr/bin/env bash
# Same 3-instance A/B as DeepSeek / homemade Meta proxy, but via
# 1rgs/claude-code-proxy (LiteLLM) → Meta openai-compat Opus.
#
# Hypothesis: if apologize/tool loops drop vs claude_llama_proxy, the homemade
# Anthropic↔OpenAI conversion was the main amplifier.
#
# Prereq: claude-code-proxy listening on PROXY (default :8082), Polar up.
#   bash examples/swebench_verified/start_claude_code_proxy_meta.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROXY="${PROXY:-http://127.0.0.1:8082}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-1800}"
# CC client id; proxy maps *opus* → BIG_MODEL=claude-4-8-opus-genai
MODEL="${MODEL:-claude-opus-4-8}"
HAIKU="${HAIKU:-claude-haiku-4-5}"

IDS=(
  psf__requests-2317
  django__django-11451
  astropy__astropy-8707
)

id_args=()
for id in "${IDS[@]}"; do
  id_args+=(--instance-id "$id")
done

if ! curl -fsS -m 3 "${PROXY%/}/" >/dev/null 2>&1 && ! curl -fsS -m 3 "${PROXY%/}/v1/messages" -X OPTIONS >/dev/null 2>&1; then
  # Health: POST smoke is done separately; just check TCP/port responds somehow
  if ! curl -fsS -m 2 -o /dev/null -w '' "${PROXY%/}/docs" 2>/dev/null; then
    # FastAPI may 404 on /; any HTTP response means up
    code=$(curl -sS -m 2 -o /dev/null -w '%{http_code}' "${PROXY%/}/" || true)
    if [[ -z "$code" || "$code" == "000" ]]; then
      echo "claude-code-proxy not reachable at $PROXY" >&2
      echo "start with: bash examples/swebench_verified/start_claude_code_proxy_meta.sh" >&2
      exit 1
    fi
  fi
fi

echo "[$(date -Is)] Opus-via-claude-code-proxy A/B arm"
echo "  proxy=$PROXY cc_model=$MODEL haiku=$HAIKU timeout=$TIMEOUT"

PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology "$TOPOLOGY" \
  --model-name "$MODEL" \
  --anthropic-base-url "$PROXY" \
  --haiku-model "$HAIKU" \
  --timeout-seconds "$TIMEOUT" \
  --export-agentreplay examples/swebench_verified/agentreplay_export \
  "${id_args[@]}" \
  2>&1 | tee /tmp/ab_opus_ccp_submit.log
