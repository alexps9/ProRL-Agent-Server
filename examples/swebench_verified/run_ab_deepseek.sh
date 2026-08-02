#!/usr/bin/env bash
# Claude Code → DeepSeek Anthropic-compatible API (deepseek-v4-pro).
# Docs: https://api-docs.deepseek.com/zh-cn/quick_start/agent_integrations/claude_code
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash examples/swebench_verified/run_ab_deepseek.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"

BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/anthropic}"
MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro[1m]}"
HAIKU="${DEEPSEEK_HAIKU_MODEL:-deepseek-v4-flash}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-1800}"

IDS=(
  psf__requests-2317
  django__django-11451
  astropy__astropy-8707
)

id_args=()
for id in "${IDS[@]}"; do
  id_args+=(--instance-id "$id")
done

echo "[$(date -Is)] DeepSeek Claude Code A/B arm"
echo "  base=$BASE_URL model=$MODEL haiku/subagent=$HAIKU timeout=$TIMEOUT"

PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology "$TOPOLOGY" \
  --model-name "$MODEL" \
  --anthropic-base-url "$BASE_URL" \
  --anthropic-api-key "$DEEPSEEK_API_KEY" \
  --haiku-model "$HAIKU" \
  --effort-level max \
  --timeout-seconds "$TIMEOUT" \
  --export-agentreplay examples/swebench_verified/agentreplay_export \
  "${id_args[@]}" \
  2>&1 | tee /tmp/ab_deepseek_submit.log
