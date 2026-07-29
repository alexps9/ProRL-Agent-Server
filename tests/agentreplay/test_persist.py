"""Tests for gateway agentreplay artifact persistence + claude_code staging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polar.agent.models import AgentRunResult, AgentSpec
from polar.agent.presets.claude_code import ClaudeCodeHarness
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest, SessionResult, SessionStatus
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trace, Trajectory


class _FakeRuntime:
    """Records exec commands; maps host session_dir to /polar/session paths."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.commands: list[str] = []
        self.spec = MagicMock(workdir="/polar/session/workspace")

    async def exec(self, command: str, **kwargs):  # noqa: ANN003
        self.commands.append(command)
        # Emulate bind-mount: run a tiny subset of shell that postprocess needs.
        # We intentionally only honor the mkdir/cp pattern from postprocess.
        if "mkdir -p" in command and "claude_projects" in command:
            dest = self.session_dir / "artifacts" / "claude_projects"
            dest.mkdir(parents=True, exist_ok=True)
            projects = self.session_dir / ".claude" / "projects"
            if projects.is_dir():
                for item in projects.iterdir():
                    target = dest / item.name
                    if item.is_dir():
                        import shutil

                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(item, target)
                    else:
                        import shutil

                        shutil.copy2(item, target)
            timing = self.session_dir / ".claude" / "tool_timing.jsonl"
            if timing.is_file():
                import shutil

                shutil.copy2(timing, dest / "tool_timing.jsonl")
        result = MagicMock()
        result.return_code = 0
        result.stdout = ""
        result.stderr = ""
        return result


def _make_manager(save_dir: Path | None) -> GatewayNodeManager:
    return GatewayNodeManager(
        node_id="node-1",
        gateway_url="http://127.0.0.1:8100",
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
        storage=MagicMock(),
        session_registry=MagicMock(),
        builders=MagicMock(),
        evaluators=MagicMock(),
        save_dir=str(save_dir) if save_dir else None,
    )


def _dispatch_request(task_id: str = "task-a", session_id: str = "ses-a") -> SessionDispatchRequest:
    return SessionDispatchRequest(
        session_id=session_id,
        task_id=task_id,
        instruction="do the thing",
        remaining_timeout_seconds=60.0,
        agent=AgentSpec(harness="claude_code", model_name="test-model"),
        builder={"strategy": "prefix_merging"},
        metadata={"benchmark": "unit"},
    )


@pytest.mark.asyncio
async def test_claude_code_postprocess_stages_projects(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    projects = session_dir / ".claude" / "projects" / "my-slug"
    projects.mkdir(parents=True)
    (projects / "sid.jsonl").write_text('{"type":"user"}\n')
    (session_dir / ".claude" / "tool_timing.jsonl").write_text('{"ts":1}\n')
    (session_dir / "artifacts").mkdir()

    harness = ClaudeCodeHarness(AgentSpec(harness="claude_code", model_name="m"))
    runtime = _FakeRuntime(session_dir)
    await harness.postprocess(
        runtime,  # type: ignore[arg-type]
        AgentRunResult(status="completed", return_code=0),
    )

    staged = session_dir / "artifacts" / "claude_projects"
    assert (staged / "my-slug" / "sid.jsonl").is_file()
    assert (staged / "tool_timing.jsonl").is_file()


@pytest.mark.asyncio
async def test_persist_agentreplay_artifacts_to_save_dir(tmp_path: Path) -> None:
    save_dir = tmp_path / "rollout_results"
    session_dir = tmp_path / "ephemeral"
    src = session_dir / "artifacts" / "claude_projects" / "slug"
    src.mkdir(parents=True)
    (src / "sess.jsonl").write_text('{"type":"user"}\n')

    manager = _make_manager(save_dir)
    request = _dispatch_request()
    managed = ManagedSession(
        request=request,
        session_dir=session_dir,
        artifacts_dir=session_dir / "artifacts",
        timer=StageTimer(),
    )
    result = SessionResult(
        session_id=request.session_id,
        task_id=request.task_id,
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(
            status="COMPLETED",
            traces=[Trace(reward=1.0)],
        ),
    )

    await manager._persist_agentreplay_artifacts(managed, result)

    dest = save_dir / f"task_{request.task_id}" / f"ses_{request.session_id}"
    assert (dest / "claude_projects" / "slug" / "sess.jsonl").is_file()
    meta = json.loads((dest / "meta.json").read_text())
    assert meta["harness"] == "claude_code"
    assert meta["reward"] == 1.0
    assert meta["metadata"]["benchmark"] == "unit"


@pytest.mark.asyncio
async def test_persist_skips_without_save_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "ephemeral"
    src = session_dir / "artifacts" / "claude_projects" / "slug"
    src.mkdir(parents=True)
    (src / "sess.jsonl").write_text("{}\n")

    manager = _make_manager(None)
    request = _dispatch_request()
    managed = ManagedSession(
        request=request,
        session_dir=session_dir,
        artifacts_dir=session_dir / "artifacts",
        timer=StageTimer(),
    )
    await manager._persist_agentreplay_artifacts(managed, None)
    # Nothing raised; no save_dir means no copy.


@pytest.mark.asyncio
async def test_end_to_end_export_smoke(tmp_path: Path) -> None:
    """Simulate harness stage → gateway persist → export CLI (no Docker/GPU)."""
    save_dir = tmp_path / "rollout_results"
    session_dir = tmp_path / "ephemeral"
    projects = session_dir / ".claude" / "projects" / "-polar-session-workspace"
    projects.mkdir(parents=True)
    (projects / "deadbeef.jsonl").write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "fix"}}) + "\n"
    )
    sub = projects / "deadbeef" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-x.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "ok"}})
        + "\n"
    )
    (session_dir / "artifacts").mkdir()

    harness = ClaudeCodeHarness(AgentSpec(harness="claude_code", model_name="m"))
    await harness.postprocess(
        _FakeRuntime(session_dir),  # type: ignore[arg-type]
        AgentRunResult(status="completed", return_code=0),
    )

    manager = _make_manager(save_dir)
    request = _dispatch_request(task_id="swe-1", session_id="sid-1")
    managed = ManagedSession(
        request=request,
        session_dir=session_dir,
        artifacts_dir=session_dir / "artifacts",
        timer=StageTimer(),
    )
    await manager._persist_agentreplay_artifacts(
        managed,
        SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status=SessionStatus.COMPLETED,
            trajectory=Trajectory(status="COMPLETED", traces=[Trace(reward=0.0)]),
        ),
    )

    out = tmp_path / "raw_from_polar"
    from polar.agentreplay import export_from_save_dir

    stats = export_from_save_dir(save_dir, out)
    assert stats["sessions"] == 1
    assert stats["main_jsonl"] == 1
    main = out / "-polar-session-workspace" / "deadbeef.jsonl"
    assert main.is_file()
    assert (out / "-polar-session-workspace" / "deadbeef" / "subagents" / "agent-x.jsonl").is_file()

    # Structural sanity (agentreplay-style): every main jsonl line is valid JSON.
    for line in main.read_text().splitlines():
        if line.strip():
            json.loads(line)
