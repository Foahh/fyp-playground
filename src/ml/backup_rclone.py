from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path


def build_dest(backup_dest_base: str, run_name: str) -> str:
    base = backup_dest_base.rstrip("/").rstrip()
    if not base:
        raise ValueError("backup_dest_base must be non-empty")
    return f"{base}/results/model/{run_name}"


def _tail(s: str, max_chars: int = 2000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[-max_chars:]


def sync_run_dir(local_run_dir: Path, dest: str, timeout_s: int) -> int:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    src = str(local_run_dir.resolve())

    cmd = [
        "rclone",
        "sync",
        "--create-empty-src-dirs",
        "--contimeout",
        "10s",
        "--timeout",
        "30s",
        "--retries",
        "1",
        f"{src}/",
        f"{dest.rstrip('/')}/",
    ]

    print(f"[backup] rclone sync start src={src} dest={dest}")
    try:
        p = subprocess.run(cmd, timeout=timeout_s, check=False, text=True, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"[backup] rclone sync timeout after {timeout_s}s")
        return 124

    print(f"[backup] rclone sync done exit_code={p.returncode}")
    if p.returncode != 0:
        stderr_tail = _tail(p.stderr)
        if stderr_tail.strip():
            print(f"[backup] rclone stderr tail:\n{stderr_tail}")
    return p.returncode


@dataclass
class BackupManager:
    local_run_dir: Path
    dest: str
    timeout_s: int = 600
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _pending: bool = field(default=False, init=False)
    _reasons: list[str] = field(default_factory=list, init=False)

    def request_sync(self, reason: str) -> None:
        self._pending = True
        if reason:
            self._reasons.append(reason)

    def maybe_run_sync(self) -> bool:
        if not self._pending:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if not self._pending:
                return False
            reasons = ", ".join(self._reasons[-5:]) if self._reasons else "unspecified"
            self._pending = False
            self._reasons.clear()
            print(f"[backup] sync requested reasons={reasons}")
            sync_run_dir(self.local_run_dir, self.dest, timeout_s=self.timeout_s)
            return True
        finally:
            self.lock.release()

