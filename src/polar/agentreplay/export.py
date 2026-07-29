"""Export Polar save_dir Claude Code transcripts into agentreplay staging layout.

Polar persists per-session artifacts as::

    {save_dir}/task_{task_id}/ses_{session_id}/claude_projects/<slug>/...
    {save_dir}/task_{task_id}/ses_{session_id}/meta.json

agentreplay ``collect`` / ``convert`` expect a flat projects tree::

    {out}/<slug>/<sessionId>.jsonl
    {out}/<slug>/<sessionId>/subagents/...

This module merges every session's ``claude_projects/`` into that layout.
Optional sidecar metadata (does not touch raw jsonl) is written under
``{out}/_polar_meta/<task_id>__<session_id>.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def export_from_save_dir(
    save_dir: str | Path,
    out: str | Path,
    *,
    include_meta: bool = True,
) -> dict[str, int]:
    """Copy staged Claude projects from *save_dir* into agentreplay *out*.

    Returns counts: ``{"sessions": N, "slugs": M, "main_jsonl": K}``.
    """
    root = Path(save_dir).expanduser().resolve()
    dest = Path(out).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"save_dir not found: {root}")
    dest.mkdir(parents=True, exist_ok=True)

    sessions = 0
    slugs: set[str] = set()
    main_jsonl = 0
    meta_dir = dest / "_polar_meta"

    for projects_dir in sorted(root.glob("task_*/ses_*/claude_projects")):
        session_root = projects_dir.parent
        task_dir = session_root.parent
        task_id = task_dir.name.removeprefix("task_")
        session_id = session_root.name.removeprefix("ses_")
        sessions += 1

        for entry in projects_dir.iterdir():
            # tool_timing.jsonl may sit at the claude_projects root. Use a
            # non-.jsonl suffix so agentreplay convert's recursive *.jsonl
            # walk does not treat timing events as session transcripts.
            if entry.name == "tool_timing.jsonl":
                if include_meta:
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        entry,
                        meta_dir / f"{task_id}__{session_id}.tool_timing.jsonl",
                    )
                continue
            if not entry.is_dir():
                continue
            slug = entry.name
            slugs.add(slug)
            slug_dest = dest / slug
            slug_dest.mkdir(parents=True, exist_ok=True)
            for item in entry.iterdir():
                target = slug_dest / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
                    if item.suffix == ".jsonl":
                        main_jsonl += 1

        if include_meta:
            meta_src = session_root / "meta.json"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_dest = meta_dir / f"{task_id}__{session_id}.json"
            if meta_src.is_file():
                shutil.copy2(meta_src, meta_dest)
            else:
                payload: dict[str, Any] = {
                    "task_id": task_id,
                    "session_id": session_id,
                }
                meta_dest.write_text(json.dumps(payload, indent=2) + "\n")

    return {
        "sessions": sessions,
        "slugs": len(slugs),
        "main_jsonl": main_jsonl,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage Polar Claude Code artifacts for agentreplay collect/convert.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Polar rollout.save_dir (contains task_*/ses_*/claude_projects/).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output staging directory (agentreplay projects layout).",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Skip writing _polar_meta sidecars.",
    )
    args = parser.parse_args(argv)
    stats = export_from_save_dir(
        args.save_dir,
        args.out,
        include_meta=not args.no_meta,
    )
    print(
        f"exported {stats['sessions']} session(s), "
        f"{stats['slugs']} slug(s), "
        f"{stats['main_jsonl']} main jsonl -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
