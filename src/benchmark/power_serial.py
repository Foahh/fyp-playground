"""INA228 serial capture: protobuf-based energy accumulator with edge-triggered windows."""

from __future__ import annotations

import struct
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tenacity import Retrying, stop_after_attempt, wait_fixed

from .constants import POWER_MEASURE_CSV_PATH


def _emit_stdout(msg: str) -> None:
    try:
        from .logutil import configure_logging, get_logger

        configure_logging()
        get_logger("power").info(msg)
    except Exception:
        print(msg)


def _emit_error(msg: str) -> None:
    try:
        from .logutil import configure_logging, get_logger

        configure_logging()
        get_logger("power").error(msg)
    except Exception:
        print(msg)


def _format_sample_log_line(sample: dict) -> str:
    # Stable, grep-friendly, single-line format.
    host = datetime.now(timezone.utc).isoformat()
    return (
        "PM_SAMPLE "
        f"host_time_iso={host} "
        f"timestamp_us={sample['timestamp_us']} "
        f"energy_j={sample['energy_j']:.9f} "
        f"duration_us={sample['duration_us']} "
        f"is_inference={int(sample['is_inference'])} "
        f"avg_mw={sample['avg_mw']:.6f}"
    )


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
        _emit_stdout(f"Auto-detected ESP32-C6 power monitor: {best_port} (score={best_score})")

    return best_port


def compute_power_metrics(samples: list[dict], num_inferences: int = 1) -> dict:
    """
    Compute power metrics from protobuf samples.

    Args:
        samples: List of power sample dictionaries
        num_inferences: Number of inference runs (for averaging)

    Returns dict with pm_avg_* keys (power measure, averaged); grouped by quantity:
    - pm_avg_inf_mW, pm_avg_idle_mW, pm_avg_delta_mW
    - pm_avg_inf_ms, pm_avg_idle_ms (window duration)
    - pm_avg_inf_mJ, pm_avg_idle_mJ (per-run scaling via num_inferences)
    """
    if not samples or num_inferences <= 0:
        return {
            "pm_avg_inf_mW": None,
            "pm_avg_idle_mW": None,
            "pm_avg_delta_mW": None,
            "pm_avg_inf_ms": None,
            "pm_avg_idle_ms": None,
            "pm_avg_inf_mJ": None,
            "pm_avg_idle_mJ": None,
        }

    inference_samples = [s for s in samples if s.get("is_inference")]
    idle_samples = [
        s
        for s in samples
        if not s.get("is_inference") and s.get("duration_us", 0) > 0
    ]

    # Compute inference metrics
    if inference_samples:
        inf_energy_j = sum(float(s["energy_j"]) for s in inference_samples)
        inf_duration_us = sum(s["duration_us"] for s in inference_samples)
        inf_duration_s = inf_duration_us * 1e-6
        avg_power_inf = (
            (inf_energy_j / inf_duration_s) * 1000.0 if inf_duration_s > 0 else None
        )
        avg_duration_ms = (inf_duration_us / num_inferences) / 1000.0 if inf_duration_us > 0 else None
        avg_energy_mj = (
            (inf_energy_j / num_inferences) * 1000.0 if inf_energy_j > 0 else None
        )
    else:
        avg_power_inf = None
        avg_duration_ms = None
        avg_energy_mj = None

    # Compute idle metrics
    if idle_samples:
        idle_energy_j = sum(float(s["energy_j"]) for s in idle_samples)
        idle_duration_us = sum(s["duration_us"] for s in idle_samples)
        idle_duration_s = idle_duration_us * 1e-6
        avg_power_idle = (
            (idle_energy_j / idle_duration_s) * 1000.0 if idle_duration_s > 0 else None
        )
        avg_idle_ms = (
            (idle_duration_us / num_inferences) / 1000.0
            if idle_duration_us > 0
            else None
        )
        avg_energy_idle_mj = (
            (idle_energy_j / num_inferences) * 1000.0
            if idle_energy_j > 0
            else None
        )
    else:
        avg_power_idle = None
        avg_idle_ms = None
        avg_energy_idle_mj = None

    avg_power_delta = (
        (avg_power_inf - avg_power_idle)
        if (avg_power_inf is not None and avg_power_idle is not None)
        else None
    )

    return {
        "pm_avg_inf_mW": avg_power_inf,
        "pm_avg_idle_mW": avg_power_idle,
        "pm_avg_delta_mW": avg_power_delta,
        "pm_avg_inf_ms": avg_duration_ms,
        "pm_avg_idle_ms": avg_idle_ms,
        "pm_avg_inf_mJ": avg_energy_mj,
        "pm_avg_idle_mJ": avg_energy_idle_mj,
    }


def compute_avg_power_mw(samples: list[dict]) -> Optional[float]:
    """Legacy function for backward compatibility."""
    return compute_power_metrics(samples)["pm_avg_inf_mW"]


class PowerMeasureSession:
    """Reads INA228 protobuf serial for the whole benchmark run; logs to power_measure.csv."""

    _CSV_HEADER = "host_time_iso,timestamp_us,energy_j,duration_us,is_inference,avg_mw\n"
    _HANDSHAKE_REQUEST = b"PM_PING\n"
    _HANDSHAKE_ACK_PREFIX = "PM_ACK"
    _HANDSHAKE_TIMEOUT_S = 2.0

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser = None
        self.effective_port: Optional[str] = None
        self._csv_fd = None
        self._validate_samples: list[dict] = []
        self._validate_lock = threading.Lock()
        self._capture_validate = False

    def start(self, port: Optional[str], baud: int) -> bool:
        if not _HAS_PROTOBUF:
            _emit_error("WARNING: Power measurement disabled - protobuf module not found (pip install protobuf)")
            return False
        try:
            import serial
        except ImportError:
            _emit_error("WARNING: Power measurement disabled - pyserial module not found (pip install pyserial)")
            return False
        if not port:
            port = _auto_detect_esp32c6()
        if not port:
            _emit_error(
                "WARNING: Power measurement disabled - could not find an ESP32-C6 power monitor serial port. "
                "If the monitor is connected via USB, ensure USB CDC (\"USC-CDC\") is enabled on the ESP32 so it "
                "appears as /dev/ttyACM* (or pass the correct device explicitly with --power-serial)."
            )
            return False

        def _open_serial() -> Any:
            ser = serial.Serial(port, baud, timeout=0.25)
            ser.reset_input_buffer()
            return ser

        try:
            retry_open = Retrying(
                stop=stop_after_attempt(3),
                wait=wait_fixed(0.35),
                reraise=True,
            )
            self._ser = retry_open(_open_serial)
            self.effective_port = str(self._ser.port)
        except Exception as e:
            _emit_error(f"WARNING: Failed to connect to power measurement serial port {port}: {e}")
            self._ser = None
            return False

        def _handshake_or_raise() -> None:
            if not self._perform_handshake():
                raise OSError("power monitor handshake timeout or no PM_ACK")

        try:
            retry_hs = Retrying(
                stop=stop_after_attempt(3),
                wait=wait_fixed(0.35),
                reraise=True,
            )
            retry_hs(_handshake_or_raise)
        except Exception:
            _emit_error(
                "WARNING: Power measurement disabled - no ACK from power monitor. "
                "Ensure the device is running firmware with PM_PING/PM_ACK handshake support."
            )
            try:
                self._ser.close()
            except Exception as e:
                _emit_error(f"WARNING: Failed to close power serial after handshake failure: {e}")
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

    def _perform_handshake(self) -> bool:
        assert self._ser is not None
        try:
            self._ser.reset_input_buffer()
            self._ser.write(self._HANDSHAKE_REQUEST)
            self._ser.flush()
        except Exception as e:
            _emit_error(f"WARNING: Handshake write failed: {e}")
            return False

        deadline = time.monotonic() + self._HANDSHAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                line = self._ser.readline()
            except Exception as e:
                _emit_error(f"WARNING: Handshake read failed: {e}")
                return False
            if not line:
                continue
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            if text.startswith(self._HANDSHAKE_ACK_PREFIX):
                _emit_stdout(f"Power monitor ACK: {text}")
                self._ser.reset_input_buffer()
                return True
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception as e:
                _emit_error(f"WARNING: Failed to close power serial: {e}")
            self._ser = None
        if self._csv_fd is not None:
            try:
                self._csv_fd.close()
            except Exception as e:
                _emit_error(f"WARNING: Failed to close power CSV file: {e}")
            self._csv_fd = None

    def _write_csv_row(self, sample: dict) -> None:
        if self._csv_fd is None:
            return
        host = datetime.now(timezone.utc).isoformat()
        self._csv_fd.write(
            f"{host},{sample['timestamp_us']},{sample['energy_j']},"
            f"{sample['duration_us']},{int(sample['is_inference'])},{sample['avg_mw']}\n"
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
                duration_us = int(sample_pb.duration_us)
                energy_j = float(sample_pb.energy_j)
                if duration_us <= 0:
                    continue
                if energy_j < 0.0:
                    continue
                duration_s = duration_us * 1e-6
                avg_mw = (energy_j / duration_s) * 1000.0
                sample = {
                    "timestamp_us": sample_pb.timestamp_us,
                    "energy_j": energy_j,
                    "avg_mw": avg_mw,
                    "duration_us": duration_us,
                    "is_inference": sample_pb.is_inference,
                }
                self._write_csv_row(sample)
                with self._validate_lock:
                    if self._capture_validate:
                        self._validate_samples.append(sample)
            except Exception as e:
                _emit_error(f"WARNING: Power serial reader loop error: {e}")
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
    sample to results/benchmark_*/power_measure.csv with host_time_iso (UTC).
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


def get_power_session_effective_port() -> Optional[str]:
    """Serial device in use when power measurement is active; None otherwise."""
    global _session
    if _session is None:
        return None
    return _session.effective_port
