"""Shared helpers for conda_setup_*.py scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[2]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def conda_env_exists(name: str) -> bool:
    try:
        subprocess.run(
            ["conda", "run", "-n", name, "python", "-c", "import sys"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def conda_create(name: str, python_spec: str) -> None:
    run(["conda", "create", "-n", name, f"python={python_spec}", "-y"])


def conda_install(env: str, *packages: str, channels: tuple[str, ...] = ()) -> None:
    cmd = ["conda", "install", "-n", env, "-y"]
    for c in channels:
        cmd.extend(["-c", c])
    cmd.extend(packages)
    run(cmd)


def conda_run(env: str, *args: str) -> None:
    run(["conda", "run", "-n", env, "--no-capture-output", *args])


def conda_prefix(env: str) -> str:
    p = subprocess.run(
        [
            "conda",
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
        conda_create(name, python_spec)


def main_guard() -> None:
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("conda not found on PATH", file=sys.stderr)
        sys.exit(1)
