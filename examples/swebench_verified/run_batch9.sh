#!/usr/bin/env bash
# Batch9: complexity-biased SWE-bench Verified via Claude→Llama proxy.
# Usage: setsid bash examples/swebench_verified/run_batch9.sh >>/tmp/submit_batch9.log 2>&1 </dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

IDS_FILE="${IDS_FILE:-examples/swebench_verified/batch9_instance_ids.txt}"
EXPORT_DIR="${EXPORT_DIR:-examples/swebench_verified/agentreplay_export}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-10800}"
MODEL="${MODEL:-claude-opus-4-8}"
PROXY="${PROXY:-http://127.0.0.1:3456}"

bash /home/songyang/agents/start_claude_llama_proxy.sh >>/tmp/claude_llama_proxy.log 2>&1 || true
curl -fsS -m 5 "${PROXY}/health" >/dev/null
curl -fsS -m 5 http://127.0.0.1:28080/health >/dev/null

mapfile -t IDS < <(grep -v '^[[:space:]]*$' "$IDS_FILE")
echo "batch9: ${#IDS[@]} tasks  timeout=${TIMEOUT}s  model=${MODEL}  $(date -Is)"
echo "export: ${EXPORT_DIR}"

ARGS=(
  --harness claude_code
  --topology "$TOPOLOGY"
  --model-name "$MODEL"
  --anthropic-base-url "$PROXY"
  --timeout-seconds "$TIMEOUT"
  --encourage-subagents
  --export-agentreplay "$EXPORT_DIR"
)
for id in "${IDS[@]}"; do
  ARGS+=(--instance-id "$id")
done

export PYTHONUNBUFFERED=1
exec uv run python -u examples/swebench_verified/submit_swebench_tasks.py "${ARGS[@]}"
