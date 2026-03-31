"""N6 on-target workflow orchestration: generate, load, validate."""

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tenacity import Retrying, stop_after_attempt, wait_fixed

from ..constants import SSD_FAMILIES
from ..paths import BenchmarkPaths, N6_WORKDIR, STEDGEAI_PATH
from ..core.models import ModelEntry
from ..utils.logutil import get_logger
from .power_serial import (
    begin_validate_capture,
    compute_power_metrics,
    end_validate_capture,
    is_power_session_active,
)


def _append_stdout_log(text: str, benchmark_log: Path) -> None:
    """Append plain text to benchmark.log."""
    benchmark_log.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_log, "a", encoding="utf-8") as fd:
        fd.write(text)
        if not text.endswith("\n"):
            fd.write("\n")


def _get_n6_scripts_dir() -> Path:
    """Return the N6_scripts directory from STEDGEAI_CORE_DIR."""
    return Path(os.environ["STEDGEAI_CORE_DIR"]) / "scripts" / "N6_scripts"


def get_stedgeai_version(benchmark_log: Path) -> str:
    """Return normalized stedgeai version string, or ``unknown`` on failure."""
    def _probe_version() -> tuple[str, str, int]:
        return _run_streaming(
            [STEDGEAI_PATH, "--version"],
            cwd=str(N6_WORKDIR),
            timeout=20,
            benchmark_log=benchmark_log,
        )

    try:
        retry_v = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.45),
            reraise=True,
        )
        out, _, rc = retry_v(_probe_version)
        if rc == 0:
            first = out.strip().splitlines()[0] if out.strip() else ""
            m = re.search(r"\bv?(\d+\.\d+\.\d+)\b", first)
            return m.group(1) if m else "unknown"
    except Exception:
        pass

    return "unknown"


def _get_st_ai_output_dir() -> Path:
    """Return the st_ai_output directory where stedgeai generate writes."""
    return N6_WORKDIR / "st_ai_output"


def _neuralart_profile() -> str:
    """Return the --st-neural-art argument with absolute path to user_neuralart.json."""
    return f"profile_O3@{_get_n6_scripts_dir() / 'user_neuralart.json'}"


def _quiet_ai_runner_logger() -> logging.Logger:
    """Logger that suppresses stm_ai_runner console/file noise (aligned with quiet ST tools)."""
    log = logging.getLogger("fyp_benchmark.stm_ai_runner")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.CRITICAL)
    log.propagate = False
    return log


def _write_n6_loader_config() -> Path:
    """Write a config_n6l.json pointing to our workdir's generated files."""
    stedgeai_dir = Path(os.environ["STEDGEAI_CORE_DIR"])
    project_path = (
        stedgeai_dir / "Projects" / "STM32N6570-DK" / "Applications" / "NPU_Validation"
    )

    config = {
        "network.c": str(_get_st_ai_output_dir() / "network.c"),
        "project_path": str(project_path),
        "project_build_conf": "N6-DK",
        "skip_external_flash_programming": False,
        "skip_ram_data_programming": False,
    }

    config_path = N6_WORKDIR / "config_n6l.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _run_streaming(
    cmd: list,
    cwd: str,
    timeout: int,
    env=None,
    log_header: str = "",
    *,
    benchmark_log: Path,
) -> tuple[str, str, int]:
    """Run a subprocess; capture stdout/stderr and append to benchmark.log."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout, stderr, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        rc = -1
        # Re-raise after logging
        benchmark_log.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_log, "a", encoding="utf-8") as f:
            if log_header:
                f.write(log_header + "\n")
            f.write(stdout)
            f.write(stderr)
        raise

    # Append to log file
    benchmark_log.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_log, "a", encoding="utf-8") as f:
        if log_header:
            f.write(log_header + "\n")
        f.write(stdout)
        f.write(stderr)

    return stdout, stderr, rc


def _step_generate(entry: ModelEntry, benchmark_log: Path) -> tuple[str, str, int]:
    """Step 1: stedgeai generate — produce C files + memory initializers."""
    model_path = entry.model_path
    if not model_path.startswith(("http://", "https://")):
        model_path = str(Path(model_path).resolve())

    output_chpos = "chfirst" if entry.family in SSD_FAMILIES else "chlast"

    cmd = [
        STEDGEAI_PATH,
        "generate",
        "--quiet",
        "--c-api",
        "st-ai",
        "--model",
        model_path,
        "--target",
        "stm32n6",
        "--st-neural-art",
        _neuralart_profile(),
        "--enable-epoch-controller",
        "--input-data-type",
        entry.input_data_type,
        "--output-data-type",
        entry.output_data_type,
        "--inputs-ch-position",
        "chlast",
        "--outputs-ch-position",
        output_chpos,
    ]

    get_logger("workflow").info(
        "Generate step started",
        step="generate",
        variant=entry.variant,
        fmt=entry.fmt,
        model_path=model_path,
        input_dtype=entry.input_data_type,
        output_dtype=entry.output_data_type,
        output_chpos=output_chpos,
    )
    t0 = time.monotonic()
    out, err, rc = _run_streaming(
        cmd,
        cwd=str(N6_WORKDIR),
        timeout=600,
        log_header=f"\n=== GENERATE | {entry.variant} | {entry.fmt} ===",
        benchmark_log=benchmark_log,
    )
    elapsed = time.monotonic() - t0
    get_logger("workflow").info(
        "Generate step completed",
        step="generate",
        variant=entry.variant,
        rc=rc,
        elapsed_s=f"{elapsed:.3f}",
    )
    return out, err, rc


def _step_load(entry: ModelEntry, benchmark_log: Path) -> tuple[str, str, int]:
    """Step 2: n6_loader.py — build, flash, and start the test app."""
    n6_dir = _get_n6_scripts_dir()
    loader = n6_dir / "n6_loader.py"
    config_path = _write_n6_loader_config()

    cmd = [
        sys.executable,
        str(loader),
        "--n6-loader-config",
        str(config_path),
    ]

    get_logger("workflow").info(
        "Load step started",
        step="load",
        variant=entry.variant,
        fmt=entry.fmt,
        config_path=str(config_path),
    )
    t0 = time.monotonic()
    out, err, rc = _run_streaming(
        cmd,
        cwd=str(n6_dir),
        timeout=600,
        log_header=f"\n=== LOAD | {entry.variant} | {entry.fmt} ===",
        benchmark_log=benchmark_log,
    )
    elapsed = time.monotonic() - t0
    get_logger("workflow").info(
        "Load step completed",
        step="load",
        variant=entry.variant,
        rc=rc,
        elapsed_s=f"{elapsed:.3f}",
    )
    return out, err, rc


def _step_validate(
    entry: ModelEntry, n_runs: int, benchmark_log: Path
) -> tuple[str, str, int]:
    """Step 3: Use ai_runner to run multiple inferences for accurate power measurement."""

    get_logger("workflow").info(
        "Validate step started",
        step="validate",
        variant=entry.variant,
        fmt=entry.fmt,
        n_runs=n_runs,
    )
    t0 = time.monotonic()

    try:
        # Import ai_runner from STEdgeAI installation
        stedgeai_scripts = Path(os.environ["STEDGEAI_CORE_DIR"]) / "scripts" / "ai_runner"
        if str(stedgeai_scripts) not in sys.path:
            sys.path.insert(0, str(stedgeai_scripts))
        from stm_ai_runner import AiRunner
    except ImportError as e:
        get_logger("workflow").error("Validate import failed", step="validate", variant=entry.variant, error=str(e))
        return "", f"Failed to import ai_runner: {e}", 1

    output_lines = []
    _append_stdout_log(
        f"\n--- VALIDATE start | {entry.variant} | {entry.fmt} | n_runs={n_runs} ---",
        benchmark_log,
    )
    try:
        output_lines.append(f"Running {n_runs} inferences using ai_runner...")
        runner = AiRunner(logger=_quiet_ai_runner_logger())
        runner.connect("serial:921600")

        if not runner.is_connected:
            err = runner.get_error()
            _append_stdout_log(f"Connect failed: {err}", benchmark_log)
            return "", f"Failed to connect to device: {err}", 1

        # Generate random input (batch_size=1)
        inputs = runner.generate_rnd_inputs(batch_size=1)

        # Run inference n_runs times
        durations = []
        for i in range(n_runs):
            _, profile = runner.invoke(inputs, mode=AiRunner.Mode.IO_ONLY, disable_pb=True)

            if profile['c_durations']:
                d_ms = profile['c_durations'][0]
                durations.append(d_ms)
            else:
                d_ms = None

            # Sleep 50ms between runs for power measurement denoising
            time.sleep(0.05)
            get_logger("workflow").info(
                "Inference run completed",
                step="validate",
                variant=entry.variant,
                run=f"{i+1}/{n_runs}",
                duration_ms=f"{d_ms:.3f}" if d_ms else None,
            )

        runner.disconnect()

        # Format output
        if durations:
            avg_ms = sum(durations) / len(durations)
            min_ms = min(durations)
            max_ms = max(durations)
            output_lines.append(f"Completed {len(durations)} inferences")
            output_lines.append(f"Inference time: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms, max={max_ms:.3f}ms")
            _append_stdout_log(
                f"VALIDATE_SUMMARY n={len(durations)} avg_ms={avg_ms:.6f} min_ms={min_ms:.6f} max_ms={max_ms:.6f}",
                benchmark_log,
            )

        elapsed = time.monotonic() - t0
        get_logger("workflow").info(
            "Validate step completed",
            step="validate",
            variant=entry.variant,
            rc=0,
            elapsed_s=f"{elapsed:.3f}",
            inferences_completed=len(durations),
            avg_ms=f"{avg_ms:.3f}" if durations else None,
        )
        return "\n".join(output_lines), "", 0

    except Exception as e:
        elapsed = time.monotonic() - t0
        _append_stdout_log(f"Validate exception: {e}", benchmark_log)
        get_logger("workflow").error(
            "Validate step failed",
            step="validate",
            variant=entry.variant,
            error=str(e),
            elapsed_s=f"{elapsed:.3f}",
        )
        return "\n".join(output_lines), f"Error during multi-inference: {e}", 1


@dataclass
class EvalResult:
    """Collects stdout/stderr from all on-target workflow steps."""

    generate_out: str = ""
    generate_err: str = ""
    generate_rc: int = 0
    load_out: str = ""
    load_err: str = ""
    load_rc: int = 0
    validate_out: str = ""
    validate_err: str = ""
    validate_rc: int = 0
    pm_avg_inf_mW: Optional[float] = None
    pm_avg_idle_mW: Optional[float] = None
    pm_avg_delta_mW: Optional[float] = None
    pm_avg_inf_ms: Optional[float] = None
    pm_avg_idle_ms: Optional[float] = None
    pm_avg_inf_mJ: Optional[float] = None
    pm_avg_idle_mJ: Optional[float] = None

    @property
    def combined_stdout(self) -> str:
        return "\n".join(
            [self.generate_out, self.load_out, self.validate_out]
        )

    @property
    def combined_stderr(self) -> str:
        return "\n".join(
            [self.generate_err, self.load_err, self.validate_err]
        )

    @property
    def failed_step(self) -> Optional[str]:
        if self.generate_rc != 0:
            return f"generate (rc={self.generate_rc})"
        if self.load_rc != 0:
            return f"load (rc={self.load_rc})"
        if self.validate_rc != 0:
            return f"validate (rc={self.validate_rc})"
        return None


def run_benchmark(
    entry: ModelEntry, validation_count: int, paths: BenchmarkPaths
) -> EvalResult:
    """Run the full doc-based 4-step benchmark workflow."""
    res = EvalResult()
    benchmark_log = paths.benchmark_log

    try:
        # Step 1: Generate
        res.generate_out, res.generate_err, res.generate_rc = _step_generate(
            entry, benchmark_log
        )
        if res.generate_rc != 0:
            return res

        # Parse and log generate metrics
        from ..io.parsing import parse_metrics
        gen_metrics = parse_metrics(res.generate_out, res.generate_err)
        get_logger("workflow").info(
            "Generate metrics extracted",
            step="generate",
            variant=entry.variant,
            internal_ram_kib=gen_metrics.get("internal_ram_kib", ""),
            external_ram_kib=gen_metrics.get("external_ram_kib", ""),
            weights_flash_kib=gen_metrics.get("weights_flash_kib", ""),
            input_buffer_kib=gen_metrics.get("input_buffer_kib", ""),
            output_buffer_kib=gen_metrics.get("output_buffer_kib", ""),
        )

        # Step 2: Build & Flash
        res.load_out, res.load_err, res.load_rc = _step_load(entry, benchmark_log)
        if res.load_rc != 0:
            return res

        time.sleep(1)

        # Step 3: Validate on device (power measurement).
        validate_t0 = time.monotonic()
        begin_validate_capture()
        validate_lines: list = []
        try:
            res.validate_out, res.validate_err, res.validate_rc = _step_validate(
                entry, validation_count, benchmark_log
            )
            get_logger("workflow").info("Validation complete", variant=entry.variant, rc=res.validate_rc)
        finally:
            validate_dt = time.monotonic() - validate_t0
            validate_lines = end_validate_capture()

        validate_header = (
            f"\n=== VALIDATE | {entry.variant} | {entry.fmt} ===\n"
            f"elapsed time (VALIDATE, wall): {validate_dt:.3f}s\n"
            f"validate rc: {res.validate_rc}\n"
            f"power samples captured (validate window): {len(validate_lines)}\n"
        )
        _append_stdout_log(validate_header, benchmark_log)
        if res.validate_out:
            _append_stdout_log(res.validate_out, benchmark_log)
        if res.validate_err:
            _append_stdout_log(res.validate_err, benchmark_log)

        val_metrics = parse_metrics(res.validate_out, res.validate_err)
        get_logger("workflow").info(
            "Validate metrics extracted",
            step="validate",
            variant=entry.variant,
            inference_time_ms=val_metrics.get("inference_time_ms", ""),
            inf_per_sec=val_metrics.get("inf_per_sec", ""),
        )

        if validate_lines:
            metrics = compute_power_metrics(validate_lines, validation_count)
            res.pm_avg_inf_mW = metrics["pm_avg_inf_mW"]
            res.pm_avg_idle_mW = metrics["pm_avg_idle_mW"]
            res.pm_avg_delta_mW = metrics["pm_avg_delta_mW"]
            res.pm_avg_inf_ms = metrics["pm_avg_inf_ms"]
            res.pm_avg_idle_ms = metrics["pm_avg_idle_ms"]
            res.pm_avg_inf_mJ = metrics["pm_avg_inf_mJ"]
            res.pm_avg_idle_mJ = metrics["pm_avg_idle_mJ"]
            get_logger("workflow").info(
                "Power metrics computed",
                step="validate",
                variant=entry.variant,
                pm_avg_inf_mW=f"{res.pm_avg_inf_mW:.3f}" if res.pm_avg_inf_mW else "",
                pm_avg_idle_mW=f"{res.pm_avg_idle_mW:.3f}" if res.pm_avg_idle_mW else "",
                pm_avg_delta_mW=f"{res.pm_avg_delta_mW:.3f}" if res.pm_avg_delta_mW else "",
                pm_avg_inf_ms=f"{res.pm_avg_inf_ms:.3f}" if res.pm_avg_inf_ms else "",
                pm_avg_idle_ms=f"{res.pm_avg_idle_ms:.3f}" if res.pm_avg_idle_ms else "",
                pm_avg_inf_mJ=f"{res.pm_avg_inf_mJ:.3f}" if res.pm_avg_inf_mJ else "",
                pm_avg_idle_mJ=f"{res.pm_avg_idle_mJ:.3f}" if res.pm_avg_idle_mJ else "",
            )
        elif is_power_session_active():
            get_logger("workflow").warning("Power active but no samples captured", variant=entry.variant)
        if res.validate_rc != 0:
            get_logger("workflow").warning("Validation failed", variant=entry.variant, rc=res.validate_rc)

    except subprocess.TimeoutExpired as e:
        step = "on-target"
        res.generate_err += f"\nTIMEOUT in {step}: {e}"
        res.generate_rc = -1
        return res

    return res
