#!/usr/bin/env python3
"""Format first-party Python and YAML (excludes external/, results/)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _skip_yaml_path(root: Path, p: Path) -> bool:
    skip_parts = {
        "external",
        "results",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".hydra",
    }
    if any(part in skip_parts for part in p.parts):
        return True
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True
    parts = rel.parts
    return len(parts) >= 2 and parts[0] == "configs" and parts[1] == "outputs"


def iter_yaml_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        for p in root.rglob(pattern):
            if _skip_yaml_path(root, p):
                continue
            out.append(p)
    return sorted(set(out))


def format_yaml_paths(paths: list[Path]) -> int:
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    errors = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.load(text)
            with path.open("w", encoding="utf-8", newline="\n") as f:
                yaml.dump(data, f)
        except Exception as e:
            print(f"{path}: {e}", file=sys.stderr)
            errors += 1
    return errors


def yaml_check(paths: list[Path]) -> int:
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    rc = 0
    for path in paths:
        original = path.read_text(encoding="utf-8")
        data = yaml.load(original)
        buf = StringIO()
        yaml.dump(data, buf)
        formatted = buf.getvalue()
        if formatted != original:
            print(f"Would reformat: {path}", file=sys.stderr)
            rc = 1
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Format Python (ruff) and YAML (ruamel.yaml)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only: ruff format --check; YAML compare without writing",
    )
    args = parser.parse_args()

    root = repo_root()
    yaml_files = iter_yaml_files(root)

    rc = 0
    py_cmd = ["ruff", "format", *(["--check"] if args.check else []), "project.py", "src"]
    print("+", " ".join(py_cmd))
    r = subprocess.run(py_cmd, cwd=root)
    if r.returncode != 0:
        rc = r.returncode

    if yaml_files:
        if args.check:
            if yaml_check(yaml_files):
                rc = 1
        else:
            err = format_yaml_paths(yaml_files)
            if err:
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
