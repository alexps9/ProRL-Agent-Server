#!/usr/bin/env python3
"""Classify Polar-exported Claude Code traces with agentreplay.callgraph.

Reads ``agentreplay_export/`` (CC projects layout + optional ``_polar_meta/``),
builds the subagent call graph, and writes a JSON + Markdown report that buckets
sessions by structure (solo / with-subagent / fan-out), length, and SWE instance.

    uv run python examples/swebench_verified/classify_agentreplay_traces.py \
        --export examples/swebench_verified/agentreplay_export \
        --out examples/swebench_verified/agentreplay_classified
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentreplay import callgraph


def _count_lines(path: Path) -> int:
    try:
        with open(path, errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _tool_counts(path: Path) -> Counter:
    tools: Counter = Counter()
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") != "assistant":
                    continue
                content = (o.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools[b.get("name") or "?"] += 1
    except OSError:
        pass
    return tools


def _load_meta(meta_dir: Path) -> dict[str, dict[str, Any]]:
    """Map main jsonl stem (session uuid) -> polar meta if present."""
    out: dict[str, dict[str, Any]] = {}
    if not meta_dir.is_dir():
        return out
    for p in meta_dir.glob("*.json"):
        try:
            data = json.load(open(p))
        except Exception:
            continue
        # filename: <task>__<session_id>.json ; session_id is sk-polar-...
        # Prefer matching via claude session uuid from export layout later.
        out[p.stem] = data
    return out


def _assistant_turns(path: Path) -> int:
    """Distinct assistant message.id count (LLM turns)."""
    seen: set[str] = set()
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") != "assistant":
                    continue
                mid = (o.get("message") or {}).get("id")
                if mid:
                    seen.add(mid)
    except OSError:
        pass
    return len(seen)


def classify_export(export_dir: Path) -> dict[str, Any]:
    projects_root = export_dir
    # Prefer the CC projects dir if present as a single slug folder.
    slug_dirs = [
        p
        for p in export_dir.iterdir()
        if p.is_dir() and p.name not in ("_polar_meta",) and not p.name.startswith(".")
    ]
    edges = callgraph.build_all(str(projects_root))
    # reverse: parent session -> children
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for child, edge in edges.items():
        children[edge["parent_program"]].append(edge)

    sessions: list[dict[str, Any]] = []
    for slug_dir in slug_dirs:
        for main in sorted(slug_dir.glob("*.jsonl")):
            sess = main.stem
            lines = _count_lines(main)
            sub_dir = main.parent / sess / "subagents"
            sub_files = list(sub_dir.rglob("*.jsonl")) if sub_dir.is_dir() else []
            sub_lines = sum(_count_lines(p) for p in sub_files)
            tools = _tool_counts(main)
            agent_calls = tools.get("Agent", 0) + tools.get("Task", 0)
            kids = children.get(sess, [])
            # fan-out: max children sharing the same spawned_at_step
            by_step: Counter = Counter()
            for e in kids:
                step = e.get("spawned_at_step")
                if step is not None:
                    by_step[step] += 1
            max_fanout = max(by_step.values()) if by_step else 0

            turns = _assistant_turns(main)
            total = lines + sub_lines
            if total < 100:
                bucket = "near_empty"
            elif kids or sub_files:
                if max_fanout >= 3:
                    bucket = "subagent_fanout_ge3"
                elif max_fanout == 2 or len(kids) >= 2 or len(sub_files) >= 2:
                    bucket = "subagent_multi"
                else:
                    bucket = "subagent_single"
            elif agent_calls > 0:
                bucket = "agent_call_no_transcript"
            elif total >= 3000:
                bucket = "solo_long"
            elif total >= 1000:
                bucket = "solo_medium"
            else:
                bucket = "solo_short"

            sessions.append(
                {
                    "session_id": sess,
                    "slug": slug_dir.name,
                    "main_lines": lines,
                    "sub_lines": sub_lines,
                    "total_lines": total,
                    "assistant_turns": turns,
                    "n_sub_files": len(sub_files),
                    "n_sub_edges": len(kids),
                    "max_fanout": max_fanout,
                    "agent_tool_calls": agent_calls,
                    "top_tools": tools.most_common(8),
                    "bucket": bucket,
                    "children": [
                        {
                            "child": e["child_program"],
                            "spawned_at_step": e.get("spawned_at_step"),
                            "link_via": e.get("link_via"),
                        }
                        for e in kids
                    ],
                }
            )

    # Enrich from _polar_meta by matching exported session via listing order is hard;
    # instead index meta by scanning main jsonl for polar isn't available.
    # Map meta session_id (sk-polar-*) is NOT the CC uuid. Keep buckets as-is.

    by_bucket = Counter(s["bucket"] for s in sessions)
    usable = [s for s in sessions if s["bucket"] != "near_empty"]
    report = {
        "export_dir": str(export_dir),
        "n_sessions": len(sessions),
        "n_usable_ge100": len(usable),
        "n_callgraph_edges": len(edges),
        "bucket_counts": dict(by_bucket),
        "usable_with_sub_edges": sum(1 for s in usable if s["n_sub_edges"] > 0),
        "usable_with_sub_files": sum(1 for s in usable if s["n_sub_files"] > 0),
        "total_lines_usable": sum(s["total_lines"] for s in usable),
        "sessions": sorted(sessions, key=lambda s: -s["total_lines"]),
    }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Agentreplay trace classification",
        "",
        f"- export: `{report['export_dir']}`",
        f"- sessions: **{report['n_sessions']}** (usable ≥100 lines: **{report['n_usable_ge100']}**)",
        f"- callgraph edges: **{report['n_callgraph_edges']}**",
        f"- usable with subagent edges: **{report['usable_with_sub_edges']}**",
        f"- usable with subagent files: **{report['usable_with_sub_files']}**",
        f"- usable total JSONL lines: **{report['total_lines_usable']}**",
        "",
        "## Buckets",
        "",
        "| Bucket | Count | Meaning |",
        "|---|---:|---|",
    ]
    meaning = {
        "near_empty": "total lines < 100 (failed/aborted early)",
        "subagent_fanout_ge3": "≥3 subagents spawned in one parent turn",
        "subagent_multi": "2+ subagent files/edges (serial or small fan-out)",
        "subagent_single": "exactly one subagent trajectory",
        "agent_call_no_transcript": "Agent/Task tool used but no subagent jsonl linked",
        "solo_long": "no subagent, ≥3000 lines",
        "solo_medium": "no subagent, 1000–2999 lines",
        "solo_short": "no subagent, 100–999 lines",
    }
    for k, v in sorted(report["bucket_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} | {meaning.get(k, '')} |")
    lines += ["", "## Top sessions", "", "| Session | Lines | Bucket | Subs | Fanout | Agent calls |", "|---|---:|---|---:|---:|---:|"]
    for s in report["sessions"][:25]:
        if s["bucket"] == "near_empty":
            continue
        lines.append(
            f"| `{s['session_id'][:8]}…` | {s['total_lines']} | {s['bucket']} | "
            f"{s['n_sub_files']} | {s['max_fanout']} | {s['agent_tool_calls']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--export",
        default="examples/swebench_verified/agentreplay_export",
        help="Directory with CC projects layout (+ optional _polar_meta)",
    )
    ap.add_argument(
        "--out",
        default="examples/swebench_verified/agentreplay_classified",
        help="Output directory for report.json + report.md",
    )
    args = ap.parse_args()
    export_dir = Path(args.export).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classify_export(export_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, out_dir / "report.md")

    # also write per-bucket session id lists for downstream convert filters
    buckets: dict[str, list[str]] = defaultdict(list)
    for s in report["sessions"]:
        buckets[s["bucket"]].append(s["session_id"])
    (out_dir / "buckets.json").write_text(json.dumps(buckets, indent=2))

    print(f"classified {report['n_sessions']} sessions -> {out_dir}")
    print("buckets:", dict(sorted(report["bucket_counts"].items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
