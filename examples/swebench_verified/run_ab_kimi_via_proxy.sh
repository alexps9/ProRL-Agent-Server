#!/usr/bin/env bash
# Same Polar+claude_code harness as DeepSeek AB, but via local Meta proxy
# forced to fireworks-kimi-k3 (isolates proxy flattening vs Meta Claude path).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROXY="${PROXY:-http://127.0.0.1:3456}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-1800}"
MODEL="${MODEL:-claude-opus-4-8}"  # CC id; proxy FORCE_MODEL rewrites upstream

IDS=(
  psf__requests-2317
  django__django-11451
  astropy__astropy-8707
)

id_args=()
for id in "${IDS[@]}"; do
  id_args+=(--instance-id "$id")
done

echo "[$(date -Is)] Kimi-via-Meta-proxy arm"
echo "  proxy=$PROXY cc_model=$MODEL (upstream forced by proxy FORCE_MODEL) timeout=$TIMEOUT"

PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py \
  --harness claude_code \
  --topology "$TOPOLOGY" \
  --model-name "$MODEL" \
  --anthropic-base-url "$PROXY" \
  --timeout-seconds "$TIMEOUT" \
  --export-agentreplay examples/swebench_verified/agentreplay_export \
  "${id_args[@]}" \
  2>&1 | tee /tmp/ab_kimi_proxy_submit.log
