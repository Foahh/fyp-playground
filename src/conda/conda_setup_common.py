"""Shared helpers for run_conda_setup_*.py scripts."""

from __future__ import annotations

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


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def conda_env_exists(name: str) -> bool:
    try:
        subprocess.run(
            [_conda_exe(), "run", "-n", name, "python", "-c", "import sys"],
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
    cmd = [_conda_exe(), "create", "-n", name, f"python={python_spec}", "-y"]
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
    cmd = [_conda_exe(), "install", "-n", env, "-y"]
    if channels:
        # Without override, conda will also hit default channels -> slower and can mix deps.
        cmd.append("--override-channels")
        for c in channels:
            cmd.extend(["-c", c])
        if strict_channel_priority:
            cmd.extend(["--strict-channel-priority"])
    cmd.extend(packages)
    run(cmd)


def conda_run(env: str, *args: str) -> None:
    run([_conda_exe(), "run", "-n", env, "--no-capture-output", *args])


def conda_prefix(env: str) -> str:
    p = subprocess.run(
        [
            _conda_exe(),
            "run",
            "-n",
            env,
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
        print(f"Using existing conda env: {name}")
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
