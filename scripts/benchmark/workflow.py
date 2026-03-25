"""N6 on-target workflow orchestration: generate, load, validate, evaluate."""

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import build_eval_config
from .constants import N6_WORKDIR, SERVICES_DIR, SSD_FAMILIES, STEDGEAI_PATH, STDOUT_LOG
from .models import ModelEntry
from .power_serial import (
    begin_validate_capture,
    compute_power_metrics,
    end_validate_capture,
    is_power_session_active,
)

def _get_n6_scripts_dir() -> Path:
    """Return the N6_scripts directory from STEDGEAI_CORE_DIR."""
    return Path(os.environ["STEDGEAI_CORE_DIR"]) / "scripts" / "N6_scripts"


def _get_st_ai_output_dir() -> Path:
    """Return the st_ai_output directory where stedgeai generate writes."""
    return N6_WORKDIR / "st_ai_output"


def _neuralart_profile() -> str:
    """Return the --st-neural-art argument with absolute path to user_neuralart.json."""
    return f"profile_O3@{_get_n6_scripts_dir() / 'user_neuralart.json'}"


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
    cmd: list, cwd: str, timeout: int, env=None, log_header: str = ""
) -> tuple[str, str, int]:
    """Run a subprocess; capture stdout/stderr in memory and append to STDOUT_LOG only."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    _log_file = open(STDOUT_LOG, "a", encoding="utf-8")
    if log_header:
        _log_file.write(log_header + "\n")
        _log_file.flush()

    def _reader(pipe, lines, log_file):
        for line in pipe:
            lines.append(line)
            log_file.write(line)
            log_file.flush()

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, stdout_lines, _log_file)
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, stderr_lines, _log_file)
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        _log_file.close()
        raise

    t_out.join()
    t_err.join()
    _log_file.close()

    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


def _step_generate(entry: ModelEntry) -> tuple[str, str, int]:
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
        "uint8",
        "--output-data-type",
        "int8",
        "--inputs-ch-position",
        "chlast",
        "--outputs-ch-position",
        output_chpos,
    ]

    return _run_streaming(
        cmd,
        cwd=str(N6_WORKDIR),
        timeout=600,
        log_header=f"\n=== GENERATE | {entry.variant} | {entry.fmt} ===",
    )


def _step_load(entry: ModelEntry) -> tuple[str, str, int]:
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

    return _run_streaming(
        cmd,
        cwd=str(n6_dir),
        timeout=600,
        log_header=f"\n=== LOAD | {entry.variant} | {entry.fmt} ===",
    )


def _step_validate(entry: ModelEntry, n_runs: int = 50) -> tuple[str, str, int]:
    """Step 3: Use ai_runner to run multiple inferences for accurate power measurement."""

    try:
        # Import ai_runner from STEdgeAI installation
        stedgeai_scripts = Path(os.environ["STEDGEAI_CORE_DIR"]) / "scripts" / "ai_runner"
        if str(stedgeai_scripts) not in sys.path:
            sys.path.insert(0, str(stedgeai_scripts))
        from stm_ai_runner import AiRunner
    except ImportError as e:
        return "", f"Failed to import ai_runner: {e}", 1

    output_lines = []
    try:
        output_lines.append(f"Running {n_runs} inferences using ai_runner...")
        runner = AiRunner(debug=False)
        runner.connect("serial:921600")

        if not runner.is_connected:
            return "", f"Failed to connect to device: {runner.get_error()}", 1

        # Generate random input (batch_size=1)
        inputs = runner.generate_rnd_inputs(batch_size=1)

        # Run inference n_runs times
        durations = []
        for i in range(n_runs):
            _, profile = runner.invoke(inputs, mode=AiRunner.Mode.IO_ONLY, disable_pb=True)

            if profile['c_durations']:
                durations.append(profile['c_durations'][0])

            # Sleep 50ms between runs for power measurement denoising
            time.sleep(0.05)

        runner.disconnect()

        # Format output
        if durations:
            avg_ms = sum(durations) / len(durations)
            min_ms = min(durations)
            max_ms = max(durations)
            output_lines.append(f"Completed {len(durations)} inferences")
            output_lines.append(f"Inference time: avg={avg_ms:.3f}ms, min={min_ms:.3f}ms, max={max_ms:.3f}ms")

        return "\n".join(output_lines), "", 0

    except Exception as e:
        return "\n".join(output_lines), f"Error during multi-inference: {e}", 1


def _step_evaluate(entry: ModelEntry) -> tuple[str, str, int]:
    """Step 4: Run model zoo evaluator (host-side) for AP metrics."""
    build_eval_config(entry)

    cmd = [
        sys.executable,
        "stm32ai_main.py",
        "--config-path",
        ".",
        "--config-name",
        "_benchmark_temp_config",
    ]

    env = os.environ.copy()
    # TODO: Does int8 need GPU evaluation?
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["HYDRA_FULL_ERROR"] = "1"
    return _run_streaming(
        cmd,
        cwd=str(SERVICES_DIR),
        timeout=3600,
        env=env,
        log_header=f"\n=== EVALUATE | {entry.variant} | {entry.fmt} ===",
    )


@dataclass
class EvalResult:
    """Collects stdout/stderr from all workflow steps."""

    generate_out: str = ""
    generate_err: str = ""
    generate_rc: int = 0
    load_out: str = ""
    load_err: str = ""
    load_rc: int = 0
    validate_out: str = ""
    validate_err: str = ""
    validate_rc: int = 0
    evaluate_out: str = ""
    evaluate_err: str = ""
    evaluate_rc: int = 0
    avg_power_inf_mW: Optional[float] = None
    avg_power_idle_mW: Optional[float] = None
    avg_power_delta_mW: Optional[float] = None
    avg_power_inf_ms: Optional[float] = None
    avg_energy_inf_mJ: Optional[float] = None

    @property
    def combined_stdout(self) -> str:
        return "\n".join(
            [self.generate_out, self.load_out, self.validate_out, self.evaluate_out]
        )

    @property
    def combined_stderr(self) -> str:
        return "\n".join(
            [self.generate_err, self.load_err, self.validate_err, self.evaluate_err]
        )

    @property
    def failed_step(self) -> Optional[str]:
        if self.generate_rc != 0:
            return f"generate (rc={self.generate_rc})"
        if self.load_rc != 0:
            return f"load (rc={self.load_rc})"
        if self.validate_rc != 0:
            return f"validate (rc={self.validate_rc})"
        if self.evaluate_rc != 0:
            return f"evaluate (rc={self.evaluate_rc})"
        return None


def run_evaluation(entry: ModelEntry, validation_count: int) -> EvalResult:
    """Run the full doc-based 4-step benchmark workflow."""
    res = EvalResult()

    try:
        # Step 1: Generate
        print("  → Generating C code...")
        res.generate_out, res.generate_err, res.generate_rc = _step_generate(entry)
        if res.generate_rc != 0:
            return res

        # Step 2: Build & Flash
        print("  → Building and flashing...")
        res.load_out, res.load_err, res.load_rc = _step_load(entry)
        if res.load_rc != 0:
            return res

        time.sleep(1)

        # Step 3: Validate on device with multiple inferences for power measurement
        print(f"  → Validating on device ({validation_count}x inferences)...")
        begin_validate_capture()
        try:
            res.validate_out, res.validate_err, res.validate_rc = _step_validate(entry, validation_count)
        finally:
            validate_lines = end_validate_capture()
        if validate_lines:
            metrics = compute_power_metrics(validate_lines, validation_count)
            res.avg_power_inf_mW = metrics["avg_power_inf_mW"]
            res.avg_power_idle_mW = metrics["avg_power_idle_mW"]
            res.avg_power_delta_mW = metrics["avg_power_delta_mW"]
            res.avg_power_inf_ms = metrics["avg_power_inf_ms"]
            res.avg_energy_inf_mJ = metrics["avg_energy_inf_mJ"]
        elif is_power_session_active():
            print("  ⚠ WARNING: Power measurement active but no samples captured during validation")

    except subprocess.TimeoutExpired as e:
        step = "on-target"
        res.generate_err += f"\nTIMEOUT in {step}: {e}"
        res.generate_rc = -1
        return res

    try:
        # Step 4: Host-side evaluation for AP metrics
        print("  → Evaluating accuracy metrics...")
        res.evaluate_out, res.evaluate_err, res.evaluate_rc = _step_evaluate(entry)
    except subprocess.TimeoutExpired:
        res.evaluate_err = "TIMEOUT: host evaluation exceeded 1 hour"
        res.evaluate_rc = -1

    return res
