#!/usr/bin/env python3
"""Real resolved/reward summary for a rollout_results_* batch.

submit_swebench_tasks.py's own console summary and each session's
meta.json read trajectory.traces[-1].reward, which the prefix_merging
builder never populates in bypass mode (--openai-base-url /
--anthropic-base-url) -- it always reports 0 resolved regardless of the
actual grading outcome. The real swebench_harness verdict lives one level
deeper, at trajectory.metadata.evaluation in each ses_*.json. This script
reads that field directly.

Usage:
    python examples/swebench_verified/analyze_batch_results.py rollout_results_codex_gpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def summarize(save_dir: Path, *, latest_only: bool = True) -> None:
    ses_files = sorted(save_dir.glob("task_*/ses_*.json"), key=lambda p: p.stat().st_mtime)
    if not ses_files:
        print(f"no ses_*.json found under {save_dir}")
        return

    if latest_only:
        # Keep only the most recent attempt per instance_id -- reruns after
        # a proxy/harness fix leave earlier failed attempts' ses_*.json
        # sitting in the same save_dir, and they'd otherwise dilute the
        # summary. Pass --all to see every attempt.
        by_instance: dict[str, Path] = {}
        for f in ses_files:
            d = _load(f)
            iid = ((d.get("trajectory") or {}).get("metadata") or {}).get("task_metadata", {}).get("instance_id", f.parent.name)
            by_instance[iid] = f  # later mtime overwrites earlier
        ses_files = sorted(by_instance.values(), key=lambda p: p.stat().st_mtime)

    rows: list[dict[str, Any]] = []
    for f in ses_files:
        d = _load(f)
        traj = d.get("trajectory") or {}
        meta = traj.get("metadata") or {}
        task_meta = meta.get("task_metadata") or {}
        eval_block = meta.get("evaluation") or {}
        report = eval_block.get("report") or {}
        grading = report.get("grading_report") or {}
        f2p = grading.get("tests_status", {}).get("FAIL_TO_PASS", {})
        # report.resolved is authoritative in all cases (empty_generation /
        # failed_apply_patch short-circuit to it without ever producing a
        # grading_report); grading_report just adds FAIL_TO_PASS detail when
        # a full test run happened.
        rows.append({
            "task_dir": f.parent.name,
            "instance_id": task_meta.get("instance_id", "?"),
            "top_status": d.get("status"),
            "top_error": d.get("error"),
            "outcome_reward": eval_block.get("outcome_reward"),
            "resolved": report.get("resolved") if "resolved" in report else None,
            "patch_applied": grading.get("patch_successfully_applied"),
            "empty_generation": report.get("empty_generation"),
            "failed_apply_patch": report.get("failed_apply_patch"),
            "fail_to_pass_ok": len(f2p.get("success", [])),
            "fail_to_pass_bad": len(f2p.get("failure", [])),
            "timing": d.get("timing") or {},
        })

    resolved = [r for r in rows if r["resolved"] is True]
    unresolved_but_ran = [r for r in rows if r["resolved"] is False]
    no_eval = [r for r in rows if r["resolved"] is None]

    print(f"{'instance_id':<38} {'resolved':<9} {'patch_ok':<9} {'F2P':<7} {'run_s':>7}  note")
    print("-" * 110)
    for r in rows:
        f2p_str = f"{r['fail_to_pass_ok']}/{r['fail_to_pass_ok'] + r['fail_to_pass_bad']}" if (r['fail_to_pass_ok'] or r['fail_to_pass_bad']) else "-"
        run_s = (r["timing"].get("run_ms") or 0) / 1000
        if r["resolved"] is None:
            note = f"{r['top_status']}: {r['top_error']}"  # crashed before/during grading
        elif r["empty_generation"]:
            note = "codex produced no patch"
        elif r["failed_apply_patch"]:
            note = "patch failed to apply"
        elif r["resolved"] is False:
            note = f"ran + graded, FAIL_TO_PASS {f2p_str}"
        else:
            note = "resolved"
        patch_ok = r["patch_applied"] if r["patch_applied"] is not None else (False if r["empty_generation"] or r["failed_apply_patch"] else "?")
        print(f"{r['instance_id']:<38} {str(r['resolved']):<9} {str(patch_ok):<9} {f2p_str:<7} {run_s:>7.0f}  {note}")

    total = len(rows)
    print("-" * 110)
    print(f"Resolved:        {len(resolved)}/{total}  ({100 * len(resolved) / total:.1f}%)")
    print(f"Ran, unresolved: {len(unresolved_but_ran)}/{total}")
    print(f"No eval report:  {len(no_eval)}/{total}  (crashed before/during grading -- check top_status/top_error)")
    print()
    print("Note: this reads trajectory.metadata.evaluation directly, not traces[-1].reward,")
    print("so it's accurate even though the console/meta.json reward fields are not (bypass-mode gap).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_dir", help="Polar rollout.save_dir for the batch, e.g. rollout_results_codex_gpt")
    parser.add_argument("--all", action="store_true", help="Show every attempt, not just the latest per instance_id")
    args = parser.parse_args()
    summarize(Path(args.save_dir), latest_only=not args.all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
