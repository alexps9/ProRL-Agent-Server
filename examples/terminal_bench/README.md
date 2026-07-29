# Terminal-Bench (Harbor) Example

Run Polar agent harnesses on [Terminal-Bench](https://www.tbench.ai) tasks
exported in Harbor layout. Each task runs an agent inside its container, then
the **`harbor`** evaluator injects `tests/`, runs `bash /tests/test.sh`, and
reads `/logs/verifier/reward.txt`.

With `--harness claude_code`, Polar also stages native Claude Code transcripts
(`projects/<slug>/*.jsonl` + subagents) into `rollout.save_dir` for
[agentreplay](https://github.com/) collect / convert.

## Prerequisites

Polar + an inference backend (vLLM or SGLang), Docker, and the Harbor CLI:

```bash
uv pip install harbor
```

## Quick Start

### 1. Pull the dataset

```bash
harbor download 'terminal-bench/terminal-bench-2-1@latest' --export --output-dir ~/terminal-bench
# alternate (newer Harbor CLI):
# harbor datasets download terminal-bench@2.0 --output-dir ~/terminal-bench
```

### 2. Build runtime images

```bash
uv run python examples/terminal_bench/build_images.py --dataset-dir ~/terminal-bench --max-tasks 10
```

### 3. Start inference + Polar

Use `topology.vllm.yaml` or `topology.sgl.yaml` (both set `rollout.save_dir: ./rollout_results`).

```bash
uv run polar serve_rollout -c examples/terminal_bench/topology.vllm.yaml
uv run polar serve_gateway -c examples/terminal_bench/topology.vllm.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/terminal_bench/topology.vllm.yaml --node-id localhost-node-02
```

### 4. Submit tasks (Claude Code + agentreplay export)

```bash
uv run python examples/terminal_bench/submit_terminal_bench_tasks.py \
  --dataset-dir ~/terminal-bench \
  --harness claude_code \
  --max-tasks 10 \
  --export-agentreplay ./data/raw_from_polar

python -m agentreplay sanity ./data/raw_from_polar
# or re-export later from save_dir:
# python -m polar.agentreplay --save-dir ./rollout_results --out ./data/raw_from_polar
```

Supported harnesses: `claude_code`, `codex`, `opencode`, `qwen_code`, `pi`,
`hermes`, `mini_swe_agent`.
