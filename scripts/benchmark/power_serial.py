"""INA228 serial capture: protobuf-based energy accumulator with edge-triggered windows."""

from __future__ import annotations

import struct
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .constants import get_power_edge_discard_ms, get_power_serial_config, POWER_MEASURE_CSV_PATH

# Add external/fyp-power-measure to path for protobuf import
_pb_path = Path(__file__).resolve().parents[2] / "external" / "fyp-power-measure"
if str(_pb_path) not in sys.path:
    sys.path.insert(0, str(_pb_path))

try:
    from power_sample_pb2 import PowerSample
    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False
    PowerSample = None


def compute_avg_power_mw(samples: list[dict]) -> Optional[float]:
    """
    Compute average power from protobuf samples with is_inference=True.

    Uses energy accumulator data: each sample has avg_mw over duration_us.
    Weighted average by duration, with optional edge trimming.
    """
    if not samples:
        return None

    inference_samples = [s for s in samples if s.get("is_inference")]
    if not inference_samples:
        all_samples = [s for s in samples if s.get("duration_us", 0) > 0]
        if not all_samples:
            return None
        total_energy = sum(s["avg_mw"] * s["duration_us"] for s in all_samples)
        total_duration = sum(s["duration_us"] for s in all_samples)
        return total_energy / total_duration if total_duration > 0 else None

    discard_start_ms, discard_end_ms = get_power_edge_discard_ms()
    if discard_start_ms == 0 and discard_end_ms == 0:
        total_energy = sum(s["avg_mw"] * s["duration_us"] for s in inference_samples)
        total_duration = sum(s["duration_us"] for s in inference_samples)
        return total_energy / total_duration if total_duration > 0 else None

    # Edge trimming: discard first/last N ms of inference windows
    discard_start_us = int(discard_start_ms * 1000)
    discard_end_us = int(discard_end_ms * 1000)

    kept_samples = []
    for s in inference_samples:
        dur = s.get("duration_us", 0)
        if dur <= (discard_start_us + discard_end_us):
            kept_samples.append(s)
            continue
        # Proportionally reduce contribution from edges
        kept_dur = dur - discard_start_us - discard_end_us
        kept_samples.append({"avg_mw": s["avg_mw"], "duration_us": kept_dur})

    if not kept_samples:
        return None
    total_energy = sum(s["avg_mw"] * s["duration_us"] for s in kept_samples)
    total_duration = sum(s["duration_us"] for s in kept_samples)
    return total_energy / total_duration if total_duration > 0 else None


class PowerMeasureSession:
    """Reads INA228 protobuf serial for the whole benchmark run; logs to power-measure.csv."""

    _CSV_HEADER = "host_time_iso,timestamp_us,avg_mw,duration_us,is_inference\n"

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser = None
        self._csv_fd = None
        self._validate_samples: list[dict] = []
        self._validate_lock = threading.Lock()
        self._capture_validate = False

    def start(self) -> bool:
        if not _HAS_PROTOBUF:
            return False
        try:
            import serial
        except ImportError:
            return False
        port, baud = get_power_serial_config()
        if not port:
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.25)
            self._ser.reset_input_buffer()
        except Exception:
            self._ser = None
            return False

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

    def _write_csv_row(self, sample: dict) -> None:
        if self._csv_fd is None:
            return
        host = datetime.now(timezone.utc).isoformat()
        self._csv_fd.write(
            f"{host},{sample['timestamp_us']},{sample['avg_mw']},"
            f"{sample['duration_us']},{int(sample['is_inference'])}\n"
        )
        self._csv_fd.flush()

    def _run(self) -> None:
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                len_bytes = self._ser.read(4)
                if len(len_bytes) != 4:
                    continue
                msg_len = struct.unpack("<I", len_bytes)[0]
                if msg_len == 0 or msg_len > 1024:
                    continue
                msg_bytes = self._ser.read(msg_len)
                if len(msg_bytes) != msg_len:
                    continue

                sample_pb = PowerSample()
                sample_pb.ParseFromString(msg_bytes)
                sample = {
                    "timestamp_us": sample_pb.timestamp_us,
                    "avg_mw": sample_pb.avg_mw,
                    "duration_us": sample_pb.duration_us,
                    "is_inference": sample_pb.is_inference,
                }
                self._write_csv_row(sample)
                with self._validate_lock:
                    if self._capture_validate:
                        self._validate_samples.append(sample)
            except Exception:
                continue

    def begin_validate_window(self) -> None:
        with self._validate_lock:
            self._validate_samples.clear()
            self._capture_validate = True

    def end_validate_window(self) -> list[dict]:
        with self._validate_lock:
            self._capture_validate = False
            return list(self._validate_samples)


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


def end_validate_capture() -> list[dict]:
    """Stop recording validate samples; return captured samples for compute_avg_power_mw."""
    global _session
    if _session is None:
        return []
    return _session.end_validate_window()
