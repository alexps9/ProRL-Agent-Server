#!/usr/bin/env bash
# Continuous batch9 pipeline:
#   - build images listed in batch9_build_ids.txt (background)
#   - whenever >= CHUNK images are ready and not yet submitted, submit them
#   - exit when build finishes AND no remaining ready unsubmitted ids
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BUILD_IDS="${BUILD_IDS:-examples/swebench_verified/batch9_build_ids.txt}"
READY_NOW="${READY_NOW:-examples/swebench_verified/batch9b_ready_ids.txt}"
EXPORT_DIR="${EXPORT_DIR:-examples/swebench_verified/agentreplay_export}"
TOPOLOGY="${TOPOLOGY:-examples/swebench_verified/topology.smoke.yaml}"
TIMEOUT="${TIMEOUT:-10800}"
MODEL="${MODEL:-claude-opus-4-8}"
PROXY="${PROXY:-http://127.0.0.1:3456}"
DONE_FILE="${DONE_FILE:-/tmp/batch9_submitted_ids.txt}"
CHUNK="${CHUNK:-12}"
BUILD_LOG="${BUILD_LOG:-/tmp/batch9_build_images.log}"
touch "$DONE_FILE"

log() { echo "[$(date -Is)] $*" | tee -a /tmp/batch9_pipeline.log; }

tag_for() {
  local id="$1"
  id="${id//\//-}"
  echo "${id//__/-}"
}

has_image() {
  docker image inspect "polar-swebench-runtime:$(tag_for "$1")" >/dev/null 2>&1
}

is_done() { grep -qxF "$1" "$DONE_FILE"; }

bash examples/swebench_verified/start_claude_llama_proxy.sh >>/tmp/claude_llama_proxy.log 2>&1 || true
curl -fsS -m 5 "${PROXY}/health" >/dev/null
curl -fsS -m 5 http://127.0.0.1:28080/health >/dev/null

# Seed wishlist = ready_now + build_ids
WISH=$(mktemp)
{
  [[ -f "$READY_NOW" ]] && cat "$READY_NOW"
  cat "$BUILD_IDS"
} | grep -v '^[[:space:]]*$' | awk '!seen[$0]++' > "$WISH"
log "wishlist $(wc -l < "$WISH") tasks"

# Start image builds in background if not already running
if ! pgrep -f 'build_images.py' >/dev/null 2>&1; then
  mapfile -t BIDS < <(grep -v '^[[:space:]]*$' "$BUILD_IDS")
  log "starting build_images for ${#BIDS[@]} ids -> $BUILD_LOG"
  build_args=()
  for id in "${BIDS[@]}"; do build_args+=(--instance-id "$id"); done
  (
    set +e
    uv run python -u examples/swebench_verified/build_images.py "${build_args[@]}"
    echo BUILD_EXIT:$? >> "$BUILD_LOG"
  ) >>"$BUILD_LOG" 2>&1 &
  echo $! > /tmp/batch9_build.pid
  log "build pid $(cat /tmp/batch9_build.pid)"
else
  log "build_images already running"
fi

submit_chunk() {
  local chunk_file="$1"
  mapfile -t IDS < <(grep -v '^[[:space:]]*$' "$chunk_file")
  ((${#IDS[@]})) || return 0
  local args=(
    --harness claude_code --topology "$TOPOLOGY" --model-name "$MODEL"
    --anthropic-base-url "$PROXY" --timeout-seconds "$TIMEOUT"
    --encourage-subagents --export-agentreplay "$EXPORT_DIR"
  )
  local id
  for id in "${IDS[@]}"; do args+=(--instance-id "$id"); done
  log "SUBMIT ${#IDS[@]}: ${IDS[*]}"
  set +e
  PYTHONUNBUFFERED=1 uv run python -u examples/swebench_verified/submit_swebench_tasks.py "${args[@]}"
  local rc=$?
  set -e
  log "submit finished rc=$rc"
  for id in "${IDS[@]}"; do echo "$id" >> "$DONE_FILE"; done
  return 0
}

# Main loop: submit ready chunks until wishlist drained and build done
idle_rounds=0
while true; do
  chunk=$(mktemp)
  while read -r id; do
    is_done "$id" && continue
    has_image "$id" || continue
    echo "$id" >> "$chunk"
    if [[ $(wc -l < "$chunk") -ge $CHUNK ]]; then
      break
    fi
  done < "$WISH"

  if [[ -s "$chunk" ]]; then
    idle_rounds=0
    submit_chunk "$chunk"
    rm -f "$chunk"
    continue
  fi
  rm -f "$chunk"

  build_alive=0
  if [[ -f /tmp/batch9_build.pid ]] && kill -0 "$(cat /tmp/batch9_build.pid)" 2>/dev/null; then
    build_alive=1
  elif pgrep -f 'examples/swebench_verified/build_images.py' >/dev/null 2>&1; then
    build_alive=1
  fi

  # count remaining not-done
  rem=0
  while read -r id; do
    is_done "$id" && continue
    rem=$((rem + 1))
  done < "$WISH"

  log "idle: remaining=$rem build_alive=$build_alive"
  if (( rem == 0 )); then
    log "all wishlist ids submitted"
    break
  fi
  if (( build_alive == 0 )); then
    idle_rounds=$((idle_rounds + 1))
    # after build ends, one more scan for stragglers
    if (( idle_rounds >= 3 )); then
      log "build done and no ready unsubmitted images left; stopping with remaining=$rem"
      break
    fi
  fi
  sleep 120
done

rm -f "$WISH"
log "pipeline complete"
