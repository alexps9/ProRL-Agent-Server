"""Resolve and enumerate a locally-exported Terminal-Bench Harbor dataset.

Pull tasks once with the Harbor CLI:

    harbor download 'terminal-bench/terminal-bench-2-1@latest' --export --output-dir <dir>

Or, with newer Harbor CLIs:

    harbor datasets download terminal-bench@2.0 --output-dir <dir>

Each task directory holds ``instruction.md``, ``task.toml``, ``environment/``
(a self-contained ``Dockerfile``) and ``tests/`` (``test.sh`` + verifier).
This module finds those directories under ``--dataset-dir``. It does not import
``harbor``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

HUB_DATASET = "terminal-bench/terminal-bench-2-1@latest"
IMAGE_PREFIX = "polar-terminal-bench"

SUPPORTED_HARNESSES = (
    "codex",
    "claude_code",
    "opencode",
    "qwen_code",
    "pi",
    "hermes",
    "mini_swe_agent",
)


@dataclass(frozen=True)
class HarborTask:
    """One Harbor task directory, with the fields Polar needs."""

    name: str
    task_dir: Path
    instruction: str
    environment_dir: Path
    tests_dir: Path
    agent_timeout: float
    verifier_timeout: float
    cpus: int | None
    memory_mb: int | None
    allow_internet: bool
    workdir: str | None


def sanitize(name: str) -> str:
    normalized = name.strip().lower().replace("__", "--")
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def env_image_for(name: str) -> str:
    return f"{IMAGE_PREFIX}-env:{sanitize(name)}"


def runtime_image_for(name: str) -> str:
    return f"{IMAGE_PREFIX}-runtime:{sanitize(name)}"


def sif_filename_for(name: str) -> str:
    return f"{sanitize(name)}.sif"


def _coerce(value: object, caster: type, default: object) -> object:
    try:
        return caster(value)  # type: ignore[call-arg]
    except (TypeError, ValueError):
        return default


def _load_task(task_dir: Path) -> HarborTask | None:
    toml_path = task_dir / "task.toml"
    tests_dir = task_dir / "tests"
    environment_dir = task_dir / "environment"
    if not (toml_path.is_file() and (tests_dir / "test.sh").is_file()):
        return None
    if not (environment_dir / "Dockerfile").is_file():
        return None

    meta = tomllib.loads(toml_path.read_text())
    agent = meta.get("agent", {})
    verifier = meta.get("verifier", {})
    env = meta.get("environment", {})
    instruction_path = task_dir / "instruction.md"
    instruction = (instruction_path.read_text() if instruction_path.is_file() else "").strip()

    return HarborTask(
        name=task_dir.name,
        task_dir=task_dir,
        instruction=instruction,
        environment_dir=environment_dir,
        tests_dir=tests_dir,
        agent_timeout=_coerce(agent.get("timeout_sec"), float, 600.0),
        verifier_timeout=_coerce(verifier.get("timeout_sec"), float, 120.0),
        cpus=_coerce(env.get("cpus"), int, None) if env.get("cpus") is not None else None,
        memory_mb=(
            _coerce(env.get("memory_mb"), int, None) if env.get("memory_mb") is not None else None
        ),
        allow_internet=bool(env.get("allow_internet", True)),
        workdir=str(env["workdir"]) if env.get("workdir") else None,
    )


def find_dataset_dir(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(
            f"Dataset dir not found: {path}\n"
            f"Pull it first with the Harbor CLI:\n"
            f"  harbor download '{HUB_DATASET}' --export --output-dir {path}"
        )
    return path


def load_tasks(
    dataset_dir: str | Path,
    *,
    max_tasks: int = -1,
    names: list[str] | None = None,
) -> list[HarborTask]:
    """Enumerate tasks under *dataset_dir* (any depth — robust to export nesting)."""
    root = find_dataset_dir(dataset_dir)
    task_dirs = sorted({p.parent for p in root.rglob("task.toml")})
    tasks = [t for t in (_load_task(d) for d in task_dirs) if t is not None]
    if not tasks:
        raise SystemExit(
            f"No tasks found under {root}. Expected per-task dirs with "
            f"task.toml + tests/test.sh + environment/Dockerfile."
        )
    if names:
        wanted = set(names)
        selected = [t for t in tasks if t.name in wanted]
        missing = sorted(wanted - {t.name for t in selected})
        if missing:
            raise SystemExit(f"Unknown task(s): {', '.join(missing)}")
        return selected
    if max_tasks > 0:
        return tasks[:max_tasks]
    return tasks
