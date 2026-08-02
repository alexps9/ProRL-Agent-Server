"""Claude Code harness — https://docs.anthropic.com/en/docs/claude-code"""

from __future__ import annotations

import json
import shlex
import textwrap

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import (
    BaseRuntime,
    RUNTIME_AGENT_LOG_DIR,
    RUNTIME_ARTIFACTS_DIR,
    RUNTIME_SESSION_DIR,
)
from polar.runtime.models import ExecInput

# Hook script written into CLAUDE_CONFIG_DIR. Uses CLAUDE_CONFIG_DIR (not
# expanduser("~/.claude")) so timing events land next to projects/ inside the
# Polar session bind-mount — required for agentreplay export.
_TOOL_TIMING_HOOK = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys, time
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    log_path = os.path.join(config_dir, "tool_timing.jsonl")
    data = json.load(sys.stdin)
    entry = {
        "ts": round(time.time(), 6),
        "event": data.get("hook_event_name", ""),
        "session_id": data.get("session_id", ""),
        "tool_name": data.get("tool_name", ""),
    }
    if data.get("tool_input"):
        entry["tool_input_keys"] = sorted(data["tool_input"].keys())
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\\n")
    """
)


class ClaudeCodeHarness(BaseHarness):
    """Run Claude Code CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Absolute path outside the workspace — $HOME won't expand in docker
        # exec -e, and a literal "$HOME" dir would get swept into git add -A.
        self._config_dir = f"{RUNTIME_SESSION_DIR}/.claude"
        self._export_agentreplay = bool(self.settings.get("export_agentreplay", True))

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._config_dir}")

        # Register MCP servers
        if self.mcp_servers:
            mcp_config: dict[str, dict] = {}
            for server in self.mcp_servers:
                entry: dict = {}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                    entry["type"] = "stdio"
                else:
                    entry["url"] = server.url
                    entry["type"] = server.transport
                mcp_config[server.name] = entry
            config = {"mcpServers": mcp_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._config_dir}/.claude.json << 'POLARCFG'\n{config_json}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._config_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._config_dir}/skills/ 2>/dev/null || true"
            )

        if self._export_agentreplay:
            await self._install_agentreplay_hooks(runtime)

    async def _install_agentreplay_hooks(self, runtime: BaseRuntime) -> None:
        """Install tool-timing hooks under CLAUDE_CONFIG_DIR for agentreplay."""
        hooks_dir = f"{self._config_dir}/hooks"
        hook_path = f"{hooks_dir}/tool_timing.py"
        await runtime.exec(f"mkdir -p {shlex.quote(hooks_dir)}")
        await runtime.exec(
            f"cat > {shlex.quote(hook_path)} << 'POLARHOOK'\n{_TOOL_TIMING_HOOK}POLARHOOK\n"
            f"chmod 755 {shlex.quote(hook_path)}"
        )

        hook_cmd = f"python3 {hook_path}"
        hook_rule = [
            {
                "matcher": "",
                "hooks": [{"type": "command", "command": hook_cmd, "async": True}],
            }
        ]
        hooks = {
            "PreToolUse": hook_rule,
            "PostToolUse": hook_rule,
            "PermissionRequest": hook_rule,
            "PermissionDenied": hook_rule,
        }

        # Merge into settings.json without wiping other keys the CLI may need.
        merge_script = textwrap.dedent(
            f"""\
            import json, os
            path = {self._config_dir!r} + "/settings.json"
            hooks = {json.dumps(hooks)}
            settings = {{}}
            if os.path.exists(path):
                with open(path) as f:
                    settings = json.load(f)
            existing = settings.get("hooks") or {{}}
            for event, rules in hooks.items():
                if event not in existing:
                    existing[event] = rules
            settings["hooks"] = existing
            with open(path, "w") as f:
                json.dump(settings, f, indent=2)
                f.write("\\n")
            """
        )
        await runtime.exec(
            f"python3 - << 'POLARMERGE'\n{merge_script}POLARMERGE"
        )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)

        flags: list[str] = [
            "--verbose",
            "--output-format=stream-json",
            "--dangerously-skip-permissions",
        ]
        for key, cli in [
            ("max_turns", "--max-turns"),
            ("reasoning_effort", "--effort"),
            ("max_budget_usd", "--max-budget-usd"),
            ("fallback_model", "--fallback-model"),
            ("append_system_prompt", "--append-system-prompt"),
            ("allowed_tools", "--allowedTools"),
            ("disallowed_tools", "--disallowedTools"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli} {shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        env: dict[str, str] = {
            **self.env,
            "CLAUDE_CONFIG_DIR": self._config_dir,
            # Allow --dangerously-skip-permissions / bypassPermissions inside
            "IS_SANDBOX": "1",
            # Suppress Statsig / telemetry calls that the CLI otherwise makes
            # to api.anthropic.com even when ANTHROPIC_BASE_URL points elsewhere.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if self.settings.get("max_thinking_tokens"):
            env["MAX_THINKING_TOKENS"] = str(self.settings["max_thinking_tokens"])

        # Model config: if model_name is set, use --model flag and pin all tier
        # aliases to the same model so claude-code doesn't try to route a
        # sub-agent / fallback request back to api.anthropic.com.
        # Aliases already present in agent.env (e.g. DeepSeek flash for haiku /
        # subagents) are preserved.
        model_flag = ""
        if self.model_name:
            model_flag = f" --model {shlex.quote(self.model_name)}"
            for alias in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                if alias not in self.env:
                    env[alias] = self.model_name
        if self.settings.get("effort_level"):
            env["CLAUDE_CODE_EFFORT_LEVEL"] = str(self.settings["effort_level"])

        return [
            ExecInput(
                command=(
                    f"claude {flags_str}{model_flag} -p {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/claude-code.txt"
                ),
                env=env,
            )
        ]

    async def postprocess(
        self, runtime: BaseRuntime, result
    ) -> None:
        """Stage Claude Code native transcripts for agentreplay before teardown."""
        if not self._export_agentreplay:
            return
        # Bind-mount makes host artifacts_dir === RUNTIME_ARTIFACTS_DIR.
        # Copy projects/ + tool_timing into artifacts/claude_projects so the
        # gateway can persist them to save_dir before session_dir is wiped.
        dest = f"{RUNTIME_ARTIFACTS_DIR}/claude_projects"
        projects = f"{self._config_dir}/projects"
        timing = f"{self._config_dir}/tool_timing.jsonl"
        await runtime.exec(
            f"mkdir -p {shlex.quote(dest)} && "
            f"if [ -d {shlex.quote(projects)} ]; then "
            f"  cp -a {shlex.quote(projects)}/. {shlex.quote(dest)}/; "
            f"fi && "
            f"if [ -f {shlex.quote(timing)} ]; then "
            f"  cp -a {shlex.quote(timing)} {shlex.quote(dest)}/tool_timing.jsonl; "
            f"fi"
        )
