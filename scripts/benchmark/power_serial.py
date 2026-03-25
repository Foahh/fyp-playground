"""INA228 serial capture: long-running session, power-measure.csv + validate-window avg."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from .constants import get_power_edge_discard_ms, get_power_serial_config, POWER_MEASURE_CSV_PATH


def parse_ina228_csv_line(line: str) -> Optional[dict]:
    """Parse one data line: ts_us,current_mA,bus_V,power_mW,inference."""
    line = line.strip()
    if not line or line.startswith("ts_us"):
        return None
    parts = line.split(",")
    if len(parts) < 4:
        return None
    try:
        ts_us = int(parts[0].strip())
    except ValueError:
        ts_us = None
    try:
        power_mw = float(parts[3])
    except ValueError:
        return None
    inference: Optional[int] = None
    if len(parts) >= 5:
        try:
            inference = int(parts[4].strip())
        except ValueError:
            inference = None
    return {"ts_us": ts_us, "power_mW": power_mw, "inference": inference}


def _contiguous_inference_one_segments(rows: list[dict]) -> list[tuple[int, int]]:
    """Inclusive index ranges [lo, hi] of rows with inference==1 in each contiguous run."""
    segments: list[tuple[int, int]] = []
    n = len(rows)
    i = 0
    while i < n:
        while i < n and rows[i].get("inference") != 1:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and rows[j].get("inference") == 1:
            j += 1
        segments.append((i, j - 1))
        i = j
    return segments


def compute_avg_power_mw(lines: list[str]) -> Optional[float]:
    """
    Prefer the mean of samples where inference==1 (inference window on STM32).

    Each contiguous inference-high segment is trimmed by discarding samples in the first
    START ms and last END ms (by ts_us); defaults are 1 ms each. Set both env vars to 0
    to disable. Reduces GPIO and power rail edge effects.

    If no inference column or no inference-high samples, use all valid samples (no edge discard).
    """
    parsed: list[dict] = []
    for line in lines:
        row = parse_ina228_csv_line(line)
        if row:
            parsed.append(row)
    if not parsed:
        return None

    discard_start_ms, discard_end_ms = get_power_edge_discard_ms()
    discard_start_us = int(round(discard_start_ms * 1000.0))
    discard_end_us = int(round(discard_end_ms * 1000.0))

    inference_samples = [r for r in parsed if r.get("inference") == 1]
    if not inference_samples:
        unsync = [r["power_mW"] for r in parsed]
        return sum(unsync) / len(unsync)

    if discard_start_us == 0 and discard_end_us == 0:
        return sum(r["power_mW"] for r in inference_samples) / len(inference_samples)

    powers_kept: list[float] = []
    for lo, hi in _contiguous_inference_one_segments(parsed):
        seg = [parsed[k] for k in range(lo, hi + 1)]
        t_first = seg[0].get("ts_us")
        t_last = seg[-1].get("ts_us")
        if t_first is None or t_last is None:
            powers_kept.extend(r["power_mW"] for r in seg)
            continue
        t_lo = t_first + discard_start_us
        t_hi = t_last - discard_end_us
        if t_hi < t_lo:
            powers_kept.extend(r["power_mW"] for r in seg)
            continue
        for r in seg:
            ts = r.get("ts_us")
            if ts is None:
                continue
            if t_lo <= ts <= t_hi:
                powers_kept.append(r["power_mW"])

    if powers_kept:
        return sum(powers_kept) / len(powers_kept)

    return sum(r["power_mW"] for r in inference_samples) / len(inference_samples)


def _skip_arduino_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    if s.startswith("ts_us"):
        return True
    if "INA228 monitor" in s or s.startswith("ERROR:"):
        return True
    return False


class PowerMeasureSession:
    """Reads INA228 serial for the whole benchmark run; logs to power-measure.csv with host time."""

    _CSV_HEADER = (
        "host_time_iso,ts_us,current_mA,bus_V,power_mW,inference\n"
    )

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser = None
        self._csv_fd = None
        self._validate_lines: list[str] = []
        self._validate_lock = threading.Lock()
        self._capture_validate = False

    def start(self) -> bool:
        try:
            import serial
        except ImportError:
            return False
        port, baud = get_power_serial_config()
        if not port:
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.25)
        except Exception:
            self._ser = None
            return False

        # Sketch waits for START before streaming; unblocks sampling + CSV header.
        try:
            self._ser.reset_input_buffer()
            self._ser.write(b"START\n")
            self._ser.flush()
        except Exception:
            pass

        path = POWER_MEASURE_CSV_PATH
        write_header = not path.exists() or path.stat().st_size == 0
        self._csv_fd = open(path, "a", encoding="utf-8", newline="")
        if write_header:
            self._csv_fd.write(self._CSV_HEADER)
            self._csv_fd.flush()

        self._thread = threading.Thread(target=self._run, name="ina228-power", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        if self._csv_fd is not None:
            try:
                self._csv_fd.close()
            except Exception:
                pass
            self._csv_fd = None

    def _write_csv_row(self, arduino_csv_line: str) -> None:
        if self._csv_fd is None:
            return
        host = datetime.now(timezone.utc).isoformat()
        self._csv_fd.write(f"{host},{arduino_csv_line}\n")
        self._csv_fd.flush()

    def _run(self) -> None:
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
            except Exception:
                break
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            tstrip = text.strip()
            if _skip_arduino_noise(tstrip):
                continue
            self._write_csv_row(tstrip)
            with self._validate_lock:
                if self._capture_validate:
                    self._validate_lines.append(tstrip)

    def begin_validate_window(self) -> None:
        with self._validate_lock:
            self._validate_lines.clear()
            self._capture_validate = True

    def end_validate_window(self) -> list[str]:
        with self._validate_lock:
            self._capture_validate = False
            return list(self._validate_lines)


_session: Optional[PowerMeasureSession] = None


def start_power_session() -> bool:
    """
    If BENCHMARK_POWER_SERIAL is set, start a background thread that logs every INA228
    sample to results/benchmark/power-measure.csv with host_time_iso (UTC).
    """
    global _session
    if _session is not None:
        return True
    port, _ = get_power_serial_config()
    if not port:
        return False
    sess = PowerMeasureSession()
    if not sess.start():
        return False
    _session = sess
    return True


def stop_power_session() -> None:
    """Stop the power logger and close the CSV (call from benchmark main finally)."""
    global _session
    if _session is not None:
        _session.stop()
        _session = None


def begin_validate_capture() -> None:
    """Clear buffer and record lines only during the next validate step."""
    global _session
    if _session is not None:
        _session.begin_validate_window()


def end_validate_capture() -> list[str]:
    """Stop recording validate lines; return captured lines for compute_avg_power_mw."""
    global _session
    if _session is None:
        return []
    return _session.end_validate_window()
