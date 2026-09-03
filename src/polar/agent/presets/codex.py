"""Codex CLI harness — https://github.com/openai/codex"""

from __future__ import annotations

import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import (
    BaseRuntime,
    RUNTIME_AGENT_LOG_DIR,
    RUNTIME_ARTIFACTS_DIR,
    RUNTIME_SESSION_DIR,
)
from polar.runtime.models import ExecInput

DEFAULT_CODEX_VERSION = "0.145.0"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_MODEL_NAME = "gpt-5.5"

# Codex's multi-agent collaboration tools (spawn_agent, followup_task,
# send_message, wait_agent, list_agents, interrupt_agent) only surface when
# the `multi_agent_v2` feature is enabled -- `multi_agent` alone (stable/true
# by default) does NOT put them in the tool list, and `tool_search` was
# removed in codex 0.145.0, so the older "call tool_search to discover them"
# path is dead (verified: an --encourage-subagents run on 0.145.0 tried
# tool_search, got "not available in the advertised tool registry", and fell
# back to plain exec). `encourage_subagents` therefore also flips on
# `--enable multi_agent_v2` (see run_steps). With v2 on, codex injects a
# "team of agents" developer prompt describing the tools; they are DIRECT
# tool calls (to=functions.collaboration.spawn_agent), never callable from
# inside functions.exec. Codex exec has no --append-system-prompt channel, so
# this nudge is prepended to the task instruction (the only slot codex reads
# before the first turn).
SUBAGENT_ENCOURAGEMENT = (
    "You lead a team of equally-capable agents. Using the collaboration tools "
    "(spawn_agent, followup_task, send_message, wait_agent, list_agents) "
    "heavily is mandatory, not optional -- call them directly, not from "
    "inside exec. Before editing code, spawn 3-5 sub-agents covering "
    "independent angles: (1) locate relevant modules/symbols, (2) find call "
    "sites and related APIs, (3) find existing tests and reproduction paths, "
    "(4) survey similar patterns elsewhere in the repo, (5) check "
    "docs/changelog/recent related commits if useful. There are 4 "
    "concurrency slots -- keep ~3 sub-agents running in parallel and "
    "wait_agent on them instead of doing the research serially yourself. "
    "After proposing a fix, spawn at least 2 more: one to verify the fix "
    "against the issue/reproduction, one to hunt for other places needing "
    "the same change or regressions. Keep spawning whenever a new "
    "independent question appears. Act via tool calls; keep narration "
    "brief.\n\n"
)


class CodexHarness(BaseHarness):
    """Run OpenAI Codex CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Keep credentials (auth.json, config.toml) outside the log dir so log
        # rotation or archival can't clobber them. Absolute path — $HOME won't
        # expand in docker exec -e.
        self._codex_home = f"{RUNTIME_SESSION_DIR}/.codex"
        self._export_agentreplay = bool(self.settings.get("export_agentreplay", True))

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._codex_home}")

        expected_version = self._expected_version()
        if expected_version:
            result = await runtime.exec(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                "if ! command -v codex >/dev/null 2>&1; then "
                "echo 'codex CLI not found; install @openai/codex in runtime.prepare' >&2; "
                "exit 127; "
                "fi; "
                "installed=$(codex --version | awk 'NF {print $NF; exit}'); "
                f"if [ \"$installed\" != {shlex.quote(expected_version)} ]; then "
                f"echo 'codex version mismatch: expected {expected_version}, got '\"$installed\" >&2; "
                "exit 1; "
                "fi"
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "codex version check failed"
                raise RuntimeError(output.strip())

        # Host-uploaded files keep the host UID, which blocks codex's
        # exec_command-based edits (cat/tee/open) on a non-root container
        # user. Other harnesses survive by rm+recreating the file. Best-effort;
        # a no-op on images without sudo.
        workdir = runtime.spec.workdir or runtime.runtime_session_dir
        await runtime.exec(
            f'sudo chown -R "$(id -u):$(id -g)" {shlex.quote(workdir)} 2>/dev/null || true'
        )

        # Register MCP servers via TOML config
        if self.mcp_servers:
            toml_lines: list[str] = []
            for server in self.mcp_servers:
                toml_lines.append(f'[mcp_servers."{server.name}"]')
                if server.transport == "stdio":
                    toml_lines.append(f'command = "{server.command}"')
                    if server.args:
                        args_str = ", ".join(f'"{a}"' for a in server.args)
                        toml_lines.append(f"args = [{args_str}]")
                else:
                    toml_lines.append(f'url = "{server.url}"')
                    toml_lines.append(f'type = "{server.transport}"')
            toml_content = "\n".join(toml_lines)
            await runtime.exec(
                f"cat > {self._codex_home}/config.toml << 'POLARCFG'\n{toml_content}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p $HOME/.agents/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* $HOME/.agents/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        encourage_subagents = bool(self.settings.get("encourage_subagents"))
        if encourage_subagents:
            instruction = SUBAGENT_ENCOURAGEMENT + instruction
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {
            **self.env,
            "CODEX_HOME": self._codex_home,
        }

        # Match Harbor's Codex harness: Codex keeps the default OpenAI provider
        # and reads the proxy from config.toml. Codex then sends Responses API
        # traffic to $OPENAI_BASE_URL/v1/responses, which Polar captures.
        flags: list[str] = [
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
        ]
        model = _cli_model_name(self.model_name)
        flags.append(f"--model {shlex.quote(model)}")
        flags.extend(["--json", "--enable unified_exec"])
        if encourage_subagents:
            # Surface the collaboration tools (spawn_agent, followup_task, ...).
            # `multi_agent` is on by default but does not add them to the tool
            # list; `multi_agent_v2` does and injects the team developer prompt.
            flags.append("--enable multi_agent_v2")

        for key, cli in [
            ("reasoning_effort", "-c model_reasoning_effort"),
            ("reasoning_summary", "-c model_reasoning_summary"),
        ]:
            value = self.settings.get(key)
            if key == "reasoning_effort" and value is None:
                value = DEFAULT_REASONING_EFFORT
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        return [
            # Write synthetic auth.json so codex picks up OPENAI_API_KEY
            ExecInput(
                command=(
                    f"mkdir -p {self._codex_home} && "
                    f'printf \'{{"OPENAI_API_KEY": "%s"}}\' "$OPENAI_API_KEY" '
                    f"> {self._codex_home}/auth.json && "
                    'if [ -n "${OPENAI_BASE_URL:-}" ]; then '
                    f"cat >> {self._codex_home}/config.toml <<POLARCODEX\n"
                    'openai_base_url = "${OPENAI_BASE_URL}"\n'
                    "POLARCODEX\n"
                    "fi"
                ),
                env=env,
            ),
            ExecInput(
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    f"codex exec {flags_str} -- {escaped} "
                    f"2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/codex.txt"
                ),
                env=env,
            ),
        ]

    def postrun_steps(self) -> list[ExecInput]:
        return [
            ExecInput(
                command=(
                    'if [ -d "$CODEX_HOME/sessions" ]; then '
                    f"rm -rf {RUNTIME_AGENT_LOG_DIR}/sessions && "
                    f'cp -R "$CODEX_HOME/sessions" {RUNTIME_AGENT_LOG_DIR}/sessions; '
                    "fi"
                ),
                env={"CODEX_HOME": self._codex_home},
            )
        ]

    async def postprocess(self, runtime: BaseRuntime, result) -> None:
        """Stage Codex native session transcripts for agentreplay before teardown.

        Mirrors ClaudeCodeHarness.postprocess: RUNTIME_ARTIFACTS_DIR is
        bind-mounted, so anything copied here is what the gateway persists to
        save_dir before the runtime's session_dir is wiped. postrun_steps()
        above only stages into RUNTIME_AGENT_LOG_DIR, which is not
        bind-mounted and never leaves the container.
        """
        if not self._export_agentreplay:
            return
        dest = f"{RUNTIME_ARTIFACTS_DIR}/codex_sessions"
        sessions = f"{self._codex_home}/sessions"
        await runtime.exec(
            f"mkdir -p {shlex.quote(dest)} && "
            f"if [ -d {shlex.quote(sessions)} ]; then "
            f"  cp -a {shlex.quote(sessions)}/. {shlex.quote(dest)}/; "
            f"fi"
        )

    def _expected_version(self) -> str | None:
        value = self.settings.get("version", DEFAULT_CODEX_VERSION)
        if value in (None, ""):
            return None
        return str(value)


def _cli_model_name(model_name: str | None) -> str:
    model = model_name or DEFAULT_MODEL_NAME
    for prefix in ("openai/", "anthropic/", "google/", "gcp/google/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model
