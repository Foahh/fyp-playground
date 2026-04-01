"""Shared helpers for run_conda_setup_*.py scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shutil import which


def repo_root() -> Path:
    """Repository root (two levels above ``src/conda/conda_setup_common.py``)."""
    return Path(__file__).resolve().parents[2]


def _conda_exe() -> str:
    """Prefer mamba when available; otherwise fall back to conda."""
    return "mamba" if which("mamba") else "conda"


def conda_cli_available() -> bool:
    """True if ``mamba`` or ``conda`` is on ``PATH``."""
    return which(_conda_exe()) is not None


def conda_prefix_base() -> Path | None:
    """Base directory for prefix-based envs (``FYP_CONDA_PREFIX_BASE``).

    If unset, prefix-based envs are disabled and named envs (``-n``) are used.
    """
    raw = os.environ.get("FYP_CONDA_PREFIX_BASE")
    if raw:
        return Path(raw).expanduser()
    return None


def conda_env_prefix(env_name: str) -> Path | None:
    base = conda_prefix_base()
    if base is None:
        return None
    return base / env_name


def conda_env_spec_args(env_name: str) -> list[str]:
    """Return ``['-p', prefix]`` (preferred) or ``['-n', name]``."""
    prefix = conda_env_prefix(env_name)
    if prefix is not None:
        return ["-p", str(prefix)]
    return ["-n", env_name]


def conda_activate_hint(env_name: str) -> str:
    """Shell snippet to activate the target env (name or prefix)."""
    prefix = conda_env_prefix(env_name)
    return f"conda activate {prefix}" if prefix is not None else f"conda activate {env_name}"


def _conda_run_cmd(env: str, *args: str) -> list[str]:
    exe = _conda_exe()
    env_args = conda_env_spec_args(env)
    if exe == "conda":
        return ["conda", "run", *env_args, "--no-capture-output", *args]
    return ["mamba", "run", *env_args, *args]


def conda_run_argv(env: str, argv: list[str]) -> list[str]:
    """Build ``conda run`` / ``mamba run`` argv so ``argv`` executes inside ``env``."""
    return _conda_run_cmd(env, *argv)


def ml_conda_env_name() -> str:
    """Training / quantization / dataset-prep env (``FYP_YOLO_ENV``, default ``fyp-ml``)."""
    return os.environ.get("FYP_YOLO_ENV", "fyp-ml")


def bhmk_conda_env_name() -> str:
    """Benchmarking / Model Zoo finetune env (``FYP_STZOO_ENV``, default ``fyp-bhmk``)."""
    return os.environ.get("FYP_STZOO_ENV", "fyp-bhmk")

def qtlz_conda_env_name() -> str:
    """Ultralytics export / INT8 TFLite quantization env (``FYP_QTLZ_ENV``, default ``fyp-qtlz``)."""
    return os.environ.get("FYP_QTLZ_ENV", "fyp-qtlz")


_BASE_ENV_COMMANDS = frozenset(
    {
        "setup-env-qtlz",
        "setup-env-ml",
        "setup-env-bhmk",
    }
)

_ML_COMMANDS = frozenset(
    {
        "download-coco",
        "download-finetune",
        "train",
    }
)

_QTLZ_COMMANDS = frozenset({"quantize"})

_BHMK_COMMANDS = frozenset(
    {
        "benchmark",
        "generate-model",
        "evaluate",
        "compare",
        "select-model",
        "prepare-finetune-dataset",
        "verify-model",
        "verify-idle-power",
        "parse-modelzoo",
        "finetune",
    }
)


def target_conda_env_for_command(command: str) -> str | None:
    """Return the conda env name ``project.py`` should run ``command`` in, or ``None``."""
    if command in _BASE_ENV_COMMANDS:
        return None
    if command in _ML_COMMANDS:
        return ml_conda_env_name()
    if command in _QTLZ_COMMANDS:
        return qtlz_conda_env_name()
    if command in _BHMK_COMMANDS:
        return bhmk_conda_env_name()
    raise ValueError(f"Unknown command for conda routing: {command!r}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def conda_env_exists(name: str) -> bool:
    try:
        env_args = conda_env_spec_args(name)
        subprocess.run(
            [_conda_exe(), "run", *env_args, "python", "-c", "import sys"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def conda_create(
    name: str,
    python_spec: str,
    *,
    channels: tuple[str, ...] = (),
    strict_channel_priority: bool = True,
) -> None:
    env_args = conda_env_spec_args(name)
    cmd = [_conda_exe(), "create", *env_args, f"python={python_spec}", "-y"]
    if channels:
        cmd.append("--override-channels")
        for c in channels:
            cmd.extend(["-c", c])
        if strict_channel_priority:
            cmd.extend(["--strict-channel-priority"])
    run(cmd)


def conda_install(
    env: str,
    *packages: str,
    channels: tuple[str, ...] = (),
    strict_channel_priority: bool = True,
) -> None:
    env_args = conda_env_spec_args(env)
    cmd = [_conda_exe(), "install", *env_args, "-y"]
    if channels:
        cmd.append("--override-channels")
        for c in channels:
            cmd.extend(["-c", c])
        if strict_channel_priority:
            cmd.extend(["--strict-channel-priority"])
    cmd.extend(packages)
    run(cmd)


def conda_run(env: str, *args: str) -> None:
    run(_conda_run_cmd(env, *args))


def conda_prefix(env: str) -> str:
    """Resolve the filesystem path of conda env ``env``.

    If ``FYP_CONDA_PREFIX_BASE`` is set, the env lives at
    ``<base>/<env>`` and that path is returned directly (same layout as
    ``conda_env_spec_args`` uses ``-p``).

    Otherwise envs are addressed by name (``-n``); ``conda`` / ``mamba run``
    injects ``CONDA_PREFIX`` inside the child process, so printing it is a
    reliable way to get the real prefix without duplicating conda's naming rules.
    """
    prefix = conda_env_prefix(env)
    if prefix is not None:
        return str(prefix)
    p = subprocess.run(
        [
            _conda_exe(),
            "run",
            *conda_env_spec_args(env),
            "python",
            "-c",
            "import os; print(os.environ['CONDA_PREFIX'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return p.stdout.strip()


def pip_install(env: str, *pip_args: str) -> None:
    conda_run(env, "pip", "install", *pip_args)


def ensure_conda_env(name: str, python_spec: str, label: str) -> None:
    if conda_env_exists(name):
        print(f"Using existing conda env: {name} ({conda_prefix(name)})")
    else:
        prefix = conda_env_prefix(name)
        if prefix is not None:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating conda env: {name} ({label}) at {prefix}")
        else:
            print(f"Creating conda env: {name} ({label})")
        # Create the base env with defaults; keep it minimal to avoid long solves.
        conda_create(name, python_spec)


def main_guard() -> None:
    try:
        subprocess.run([_conda_exe(), "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("conda/mamba not found on PATH", file=sys.stderr)
        sys.exit(1)
