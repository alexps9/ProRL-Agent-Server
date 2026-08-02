#!/usr/bin/env bash
# Small A/B: qwen_code (local Qwen via Polar gateway) vs claude_code (Meta llama proxy).
# Same instances / timeout; compare tool-loop / apolog behaviour afterwards.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROXY="${PROXY:-http://127.0.0.1:3456}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-1800}"
# Default to a tool-enabled local vLLM (see start notes in script header).
QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8011/v1}"
QWEN_MODEL="${QWEN_MODEL:-Qwen2.5-Coder-7B-Instruct}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-8}"
ARM="${1:-both}" # qwen | claude | both

IDS=(
  psf__requests-2317
  django__django-11451
  astropy__astropy-8707
)

id_args=()
for id in "${IDS[@]}"; do
  id_args+=(--instance-id "$id")
done

common=(
  --topology "$TOPOLOGY"
  --timeout-seconds "$TIMEOUT"
  "${id_args[@]}"
)

run_qwen() {
  echo "[$(date -Is)] ARM=qwen model=$QWEN_MODEL base=$QWEN_BASE_URL timeout=$TIMEOUT"
  PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py \
    --harness qwen_code \
    --model-name "$QWEN_MODEL" \
    --openai-base-url "$QWEN_BASE_URL" \
    "${common[@]}" \
    2>&1 | tee /tmp/ab_qwen_submit.log
}

run_claude() {
  echo "[$(date -Is)] ARM=claude model=$CLAUDE_MODEL proxy=$PROXY timeout=$TIMEOUT"
  # No --encourage-subagents: keep prompts closer to qwen's default agent behaviour.
  PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py \
    --harness claude_code \
    --model-name "$CLAUDE_MODEL" \
    --anthropic-base-url "$PROXY" \
    --export-agentreplay examples/swebench_verified/agentreplay_export \
    "${common[@]}" \
    2>&1 | tee /tmp/ab_claude_submit.log
}

case "$ARM" in
  qwen) run_qwen ;;
  claude) run_claude ;;
  both)
    run_qwen &
    qpid=$!
    run_claude &
    cpid=$!
    wait "$qpid"
    qrc=$?
    wait "$cpid"
    crc=$?
    echo "[$(date -Is)] done qwen_rc=$qrc claude_rc=$crc"
    exit $(( qrc != 0 || crc != 0 ? 1 : 0 ))
    ;;
  *)
    echo "usage: $0 [qwen|claude|both]" >&2
    exit 2
    ;;
esac
