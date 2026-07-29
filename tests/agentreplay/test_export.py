"""Unit tests for polar.agentreplay.export."""

from __future__ import annotations

import json
from pathlib import Path

from polar.agentreplay.export import export_from_save_dir


def _write_session(
    save_dir: Path,
    *,
    task_id: str,
    session_id: str,
    slug: str,
    main_name: str = "abc123.jsonl",
    with_subagent: bool = True,
) -> None:
    root = save_dir / f"task_{task_id}" / f"ses_{session_id}"
    projects = root / "claude_projects" / slug
    projects.mkdir(parents=True)
    (projects / main_name).write_text('{"type":"user","message":"hi"}\n')
    if with_subagent:
        sub = projects / main_name.removesuffix(".jsonl") / "subagents"
        sub.mkdir(parents=True)
        (sub / "agent-1.jsonl").write_text('{"type":"assistant"}\n')
    (root / "claude_projects" / "tool_timing.jsonl").write_text(
        '{"ts":1.0,"event":"PreToolUse"}\n'
    )
    (root / "meta.json").write_text(
        json.dumps({"task_id": task_id, "session_id": session_id, "reward": 1.0}) + "\n"
    )


def test_export_from_save_dir_merges_projects(tmp_path: Path) -> None:
    save_dir = tmp_path / "rollout_results"
    out = tmp_path / "raw"
    _write_session(save_dir, task_id="t1", session_id="s1", slug="workspace")
    _write_session(
        save_dir,
        task_id="t2",
        session_id="s2",
        slug="workspace",
        main_name="def456.jsonl",
    )

    stats = export_from_save_dir(save_dir, out)

    assert stats["sessions"] == 2
    assert stats["slugs"] == 1
    assert stats["main_jsonl"] == 2
    assert (out / "workspace" / "abc123.jsonl").is_file()
    assert (out / "workspace" / "def456.jsonl").is_file()
    assert (out / "workspace" / "abc123" / "subagents" / "agent-1.jsonl").is_file()
    assert (out / "_polar_meta" / "t1__s1.tool_timing.jsonl").is_file()
    assert (out / "_polar_meta" / "t1__s1.json").is_file()
    meta = json.loads((out / "_polar_meta" / "t1__s1.json").read_text())
    assert meta["reward"] == 1.0


def test_export_cli_main(tmp_path: Path) -> None:
    from polar.agentreplay.export import main

    save_dir = tmp_path / "rollout_results"
    out = tmp_path / "raw"
    _write_session(save_dir, task_id="t1", session_id="s1", slug="proj", with_subagent=False)
    assert main(["--save-dir", str(save_dir), "--out", str(out)]) == 0
    assert (out / "proj" / "abc123.jsonl").is_file()
