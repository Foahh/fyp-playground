"""INA228 serial capture: protobuf-based energy accumulator with edge-triggered windows."""

from __future__ import annotations

import struct
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .constants import POWER_MEASURE_CSV_PATH

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

try:
    from serial.tools import list_ports
    _HAS_SERIAL_TOOLS = True
except ImportError:
    _HAS_SERIAL_TOOLS = False


def _auto_detect_esp32c6() -> Optional[str]:
    """Auto-detect ESP32-C6 power monitor port."""
    if not _HAS_SERIAL_TOOLS:
        return None

    ports = [p for p in list_ports.comports() if p.vid is not None]
    if not ports:
        return None

    _ESPRESSIF_VIDS = {0x303A}
    best_port = None
    best_score = 0

    for port in ports:
        if port.vid not in _ESPRESSIF_VIDS:
            continue

        score = 0
        text = " ".join([
            (port.manufacturer or ""),
            (port.product or ""),
            (port.description or ""),
        ]).lower()

        if "esp32c6" in text:
            score += 8
        elif "c6" in text:
            score += 7
        elif "esp32" in text:
            score += 6
        elif "espressif" in text:
            score += 6
        elif "esp" in text:
            score += 4

        if score > best_score:
            best_score = score
            best_port = port.device

    if best_port and best_score > 0:
        print(f"Auto-detected ESP32-C6 power monitor: {best_port} (score={best_score})")

    return best_port


def compute_power_metrics(samples: list[dict], num_inferences: int = 1) -> dict:
    """
    Compute power metrics from protobuf samples.

    Args:
        samples: List of power sample dictionaries
        num_inferences: Number of inference runs (for averaging)

    Returns dict with:
    - avg_power_inf_mW: weighted average power during inference (mW)
    - avg_power_inf_ms: average duration per inference run (ms)
    - avg_energy_inf_mJ: average energy per inference run (mJ)
    """
    if not samples or num_inferences <= 0:
        return {
            "avg_power_inf_mW": None,
            "avg_power_inf_ms": None,
            "avg_energy_inf_mJ": None,
        }

    inference_samples = [s for s in samples if s.get("is_inference")]

    # Compute inference metrics
    if inference_samples:
        inf_energy_uw_us = sum(s["avg_mw"] * s["duration_us"] for s in inference_samples)
        inf_duration_us = sum(s["duration_us"] for s in inference_samples)
        avg_power_inf = inf_energy_uw_us / inf_duration_us if inf_duration_us > 0 else None
        avg_duration_ms = (inf_duration_us / num_inferences) / 1000.0 if inf_duration_us > 0 else None
        avg_energy_mj = (inf_energy_uw_us / num_inferences) / 1000.0 if inf_energy_uw_us > 0 else None
    else:
        avg_power_inf = None
        avg_duration_ms = None
        avg_energy_mj = None

    return {
        "avg_power_inf_mW": avg_power_inf,
        "avg_power_inf_ms": avg_duration_ms,
        "avg_energy_inf_mJ": avg_energy_mj,
    }


def compute_avg_power_mw(samples: list[dict]) -> Optional[float]:
    """Legacy function for backward compatibility."""
    return compute_power_metrics(samples)["avg_power_inf_mW"]


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

    def start(self, port: Optional[str], baud: int) -> bool:
        if not _HAS_PROTOBUF:
            print("WARNING: Power measurement disabled - protobuf module not found (pip install protobuf)")
            return False
        try:
            import serial
        except ImportError:
            print("WARNING: Power measurement disabled - pyserial module not found (pip install pyserial)")
            return False
        if not port:
            port = _auto_detect_esp32c6()
        if not port:
            print(
                "WARNING: Power measurement disabled - could not find an ESP32-C6 power monitor serial port. "
                "If the monitor is connected via USB, ensure USB CDC (\"USC-CDC\") is enabled on the ESP32 so it "
                "appears as /dev/ttyACM* (or pass the correct device explicitly with --power-serial)."
            )
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.25)
            self._ser.reset_input_buffer()
        except Exception as e:
            print(f"WARNING: Failed to connect to power measurement serial port {port}: {e}")
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


def start_power_session(port: Optional[str], baud: int) -> bool:
    """
    If port is provided, start a background thread that logs every INA228
    sample to results/benchmark/power-measure.csv with host_time_iso (UTC).
    """
    global _session
    if _session is not None:
        return True
    if not port:
        port = _auto_detect_esp32c6()
    sess = PowerMeasureSession()
    if not sess.start(port, baud):
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


def is_power_session_active() -> bool:
    """Check if power measurement session is currently active."""
    global _session
    return _session is not None
