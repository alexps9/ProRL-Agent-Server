#!/usr/bin/env python3
"""Compare tool-loop / apolog signals for a recent A/B batch in rollout_results_smoke."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


APOL_RE = re.compile(r"apolog|repeated drafting|rendering glitch|repetitive output", re.I)


def _task_dirs(save_dir: Path, harness: str, instance_slug: str) -> list[Path]:
    # task ids look like swebench-{harness}-{slug}-{batch}
    pat = f"task_*{harness}*{instance_slug}*"
    return sorted(save_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)


def _score_claude_export(export_root: Path) -> dict | None:
    # Prefer newest session among mains; caller passes specific session if known.
    return None


def score_agent_log(text: str) -> dict:
    lines = text.splitlines()
    apol = sum(1 for ln in lines if APOL_RE.search(ln))
    # crude repeated-line rate
    ctr = Counter(ln.strip() for ln in lines if len(ln.strip()) > 40)
    top_n = ctr.most_common(1)[0][1] if ctr else 0
    return {
        "log_lines": len(lines),
        "apol_lines": apol,
        "max_dup_line": top_n,
        "log_chars": len(text),
    }


def score_cc_jsonl(path: Path) -> dict:
    tool_sigs: Counter[tuple[str, str]] = Counter()
    asst_text = apol = 0
    for line in path.open(errors="ignore"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "assistant":
            continue
        c = (o.get("message") or {}).get("content")
        texts: list[str] = []
        if isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    texts.append(b["text"])
                elif b.get("type") == "tool_use":
                    sig = (
                        b.get("name") or "",
                        json.dumps(b.get("input", {}), sort_keys=True)[:200],
                    )
                    tool_sigs[sig] += 1
        blob = "\n".join(texts)
        if blob.strip():
            asst_text += 1
            if APOL_RE.search(blob):
                apol += 1
    max_same = tool_sigs.most_common(1)[0][1] if tool_sigs else 0
    top = tool_sigs.most_common(1)[0][0] if tool_sigs else ("", "")
    return {
        "asst_text_turns": asst_text,
        "apol_turns": apol,
        "apol_rate": apol / max(asst_text, 1),
        "n_tools": sum(tool_sigs.values()),
        "max_same_tool": max_same,
        "top_tool": top[0],
        "top_tool_args": top[1][:80],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--save-dir",
        default="rollout_results_smoke",
        type=Path,
    )
    ap.add_argument(
        "--instances",
        nargs="*",
        default=["psf-requests-2317", "django-django-11451", "astropy-astropy-8707"],
    )
    ap.add_argument("--since-minutes", type=float, default=180.0)
    args = ap.parse_args()
    save_dir = args.save_dir
    if not save_dir.is_absolute():
        save_dir = Path(__file__).resolve().parents[2] / save_dir

    import time

    cutoff = time.time() - args.since_minutes * 60
    print(f"save_dir={save_dir}")
    for harness in ("qwen_code", "claude_code"):
        print(f"\n=== {harness} ===")
        for slug in args.instances:
            dirs = [
                p
                for p in _task_dirs(save_dir, harness, slug)
                if p.stat().st_mtime >= cutoff
            ]
            if not dirs:
                print(f"  {slug}: (no recent task dir)")
                continue
            d = dirs[0]
            # agent logs
            qlog = list(d.rglob("qwen-code.txt"))
            clog = list(d.rglob("claude-code.txt"))
            projects = list(d.rglob("*.jsonl"))
            # prefer claude projects transcripts under claude_projects
            cc = [
                p
                for p in projects
                if "claude_projects" in str(p) and p.name != "agentreplay"
            ]
            print(f"  {slug}: {d.name}")
            if qlog:
                s = score_agent_log(qlog[0].read_text(errors="ignore"))
                print(f"    qwen-log: {s}")
            if clog:
                s = score_agent_log(clog[0].read_text(errors="ignore")[-2_000_000:])
                print(f"    claude-log: lines~{s['log_lines']} apol_lines={s['apol_lines']} max_dup_line={s['max_dup_line']}")
            # session jsonl in export may be outside; also try in-task
            for p in sorted(d.rglob("*.jsonl"))[:3]:
                if p.stat().st_size < 1000:
                    continue
                try:
                    s = score_cc_jsonl(p)
                except Exception as e:
                    print(f"    jsonl {p.name}: err {e}")
                    continue
                if s["n_tools"] or s["asst_text_turns"]:
                    print(f"    jsonl {p.relative_to(d)}: {s}")


if __name__ == "__main__":
    main()
