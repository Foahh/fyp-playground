"""Parallel INA228 CSV capture (Arduino) and avg power for the validate step."""

from __future__ import annotations

import os
import threading
from typing import Optional

from .constants import get_power_serial_config


def parse_ina228_csv_line(line: str) -> Optional[dict]:
    """Parse one data line from power-measure.ino (with optional trailing sync column)."""
    line = line.strip()
    if not line or line.startswith("ts_us"):
        return None
    parts = line.split(",")
    if len(parts) < 8:
        return None
    try:
        power_mw = float(parts[4])
    except ValueError:
        return None
    sync: Optional[int] = None
    if len(parts) >= 9:
        try:
            sync = int(parts[8].strip())
        except ValueError:
            sync = None
    return {"power_mW": power_mw, "sync": sync}


def compute_avg_power_mw(lines: list[str]) -> Optional[float]:
    """
    Prefer the mean of samples where sync==1 (inference window on STM32).
    If no sync column or no sync-high samples, use all valid samples.
    """
    parsed: list[dict] = []
    for line in lines:
        row = parse_ina228_csv_line(line)
        if row:
            parsed.append(row)
    if not parsed:
        return None

    synced = [r["power_mW"] for r in parsed if r.get("sync") == 1]
    if synced:
        return sum(synced) / len(synced)

    unsync = [r["power_mW"] for r in parsed]
    return sum(unsync) / len(unsync)


def _serial_reader(port: str, baud: int, lines: list[str], stop: threading.Event) -> None:
    try:
        import serial
    except ImportError:
        return
    try:
        ser = serial.Serial(port, baud, timeout=0.25)
    except Exception:
        return
    try:
        while not stop.is_set():
            raw = ser.readline()
            if raw:
                lines.append(raw.decode("utf-8", errors="replace"))
    finally:
        try:
            ser.close()
        except Exception:
            pass


def start_background_capture() -> tuple[Optional[list[str]], Optional[threading.Event], Optional[threading.Thread]]:
    """If BENCHMARK_POWER_SERIAL is set, start a thread that appends INA228 CSV lines."""
    port, baud = get_power_serial_config()
    if not port:
        return None, None, None
    lines: list[str] = []
    stop = threading.Event()
    t = threading.Thread(
        target=_serial_reader,
        args=(port, baud, lines, stop),
        name="ina228-power",
        daemon=True,
    )
    t.start()
    return lines, stop, t


def stop_capture(
    stop: Optional[threading.Event],
    thread: Optional[threading.Thread],
    join_s: float = 15.0,
) -> None:
    if stop is not None:
        stop.set()
    if thread is not None:
        thread.join(timeout=join_s)
