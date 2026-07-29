# Agentreplay export

Stage Claude Code native transcripts persisted by Polar into the layout
expected by [agentreplay](https://github.com/) `collect` / `convert` /
`sanity`.

## Layout

Polar gateway (with `rollout.save_dir` set) writes:

```text
{save_dir}/task_{task_id}/ses_{session_id}/claude_projects/<slug>/...
{save_dir}/task_{task_id}/ses_{session_id}/meta.json
```

Export merges those into:

```text
{out}/<slug>/<sessionId>.jsonl
{out}/<slug>/<sessionId>/subagents/...
{out}/_polar_meta/<task_id>__<session_id>.json
{out}/_polar_meta/<task_id>__<session_id>.tool_timing.jsonl
```

## Usage

```bash
python -m polar.agentreplay --save-dir ./rollout_results --out ./data/raw_from_polar
python -m agentreplay sanity ./data/raw_from_polar
```

The `claude_code` harness installs tool-timing hooks and copies
`$CLAUDE_CONFIG_DIR/projects` into `artifacts/claude_projects/` during
`postprocess` when `agent.settings.export_agentreplay` is true (default).
