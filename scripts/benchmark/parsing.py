"""Output and log parsing helpers for extracting metrics."""

import json
import os
import re
from pathlib import Path
from typing import Optional

from .constants import SERVICES_DIR


def _find_network_c_info() -> Optional[Path]:
    """Locate network_c_info.json from the stedgeai generate output."""
    candidates = []
    stedgeai_dir = os.environ.get("STEDGEAI_CORE_DIR", "")
    if stedgeai_dir:
        candidates.append(
            Path(stedgeai_dir)
            / "scripts"
            / "N6_scripts"
            / "st_ai_output"
            / "network_c_info.json"
        )
    # stedgeai also writes to cwd/st_ai_output by default
    candidates.append(Path("st_ai_output") / "network_c_info.json")
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_hydra_output_dir(stdout: str) -> Optional[Path]:
    """Try to find the Hydra output directory from stm32ai_main.py output."""
    m = re.search(
        r"experiments_outputs[/\\](\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", stdout
    )
    if m:
        for framework in ("tf", "pt"):
            candidate = (
                SERVICES_DIR / framework / "src" / "experiments_outputs" / m.group(1)
            )
            if candidate.is_dir():
                return candidate
    return None


def _parse_network_c_info(json_path: Path, metrics: dict):
    """Parse network_c_info.json for memory footprint metrics."""
    try:
        with open(json_path, "r") as f:
            cinfo = json.load(f)
        mem = cinfo.get("memory_footprint", {})
        if not metrics["internal_ram_kib"]:
            act = mem.get("activations", 0)
            if act:
                metrics["internal_ram_kib"] = f"{act / 1024:.2f}"
        if not metrics["weights_flash_kib"]:
            w = mem.get("weights", 0)
            if w:
                metrics["weights_flash_kib"] = f"{w / 1024:.2f}"
    except Exception:
        pass


def _parse_hydra_log(hydra_dir: Path, metrics: dict):
    """Parse stm32ai_main.log from the Hydra output directory."""
    log_path = hydra_dir / "stm32ai_main.log"
    if not log_path.exists():
        return
    try:
        log_text = log_path.read_text(encoding="utf-8")
        if not metrics["internal_ram_kib"]:
            m = re.search(
                r"activations \(rw\)\s*:\s*[\d,]+\s*B\s*\(([\d.]+)\s*KiB\)", log_text
            )
            if m:
                metrics["internal_ram_kib"] = m.group(1)
        if not metrics["weights_flash_kib"]:
            m = re.search(
                r"weights \(ro\)\s*:\s*[\d,]+\s*B\s*\(([\d.]+)\s*KiB\)", log_text
            )
            if m:
                metrics["weights_flash_kib"] = m.group(1)
        if not metrics["inference_time_ms"]:
            m = re.search(r"duration\s*:\s*([\d.]+)\s*ms", log_text)
            if m:
                metrics["inference_time_ms"] = m.group(1)
        if not metrics["external_ram_kib"]:
            m = re.search(r"hyperRAM\s+\[[^\]]+\]:\s+([\d.]+)\s+kB\s*/", log_text)
            if m and float(m.group(1)) > 0:
                metrics["external_ram_kib"] = m.group(1)
    except Exception:
        pass


def parse_metrics(stdout: str, stderr: str) -> dict:
    """Extract ap_50 and memory/inference metrics from output."""
    metrics = {
        "ap_50": "",
        "internal_ram_kib": "",
        "external_ram_kib": "",
        "weights_flash_kib": "",
        "inference_time_ms": "",
        "inf_per_sec": "",
    }

    combined = stdout + "\n" + stderr

    # ── AP@50 ──
    # 1) pycocotools format (PT COCO/SSD evaluators)
    m = re.search(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([\d.]+)",
        combined,
    )
    if m:
        metrics["ap_50"] = f"{float(m.group(1)) * 100:.2f}"

    # 2) TF keras_evaluator format
    if not metrics["ap_50"]:
        m = re.search(r"Mean AP \(mAP\):\s+([\d.]+)", combined)
        if m:
            metrics["ap_50"] = m.group(1)

    # 3) PT YOLOD trainer log format
    if not metrics["ap_50"]:
        m = re.search(r"AP50\s*:\s*([\d.]+)", combined)
        if m:
            val = float(m.group(1))
            metrics["ap_50"] = f"{val * 100:.2f}" if val <= 1.0 else f"{val:.2f}"

    # ── Memory from stedgeai generate summary ──
    m = re.search(
        r"activations \(rw\)\s*:\s*([\d,]+)\s*B\s*\(([\d.]+)\s*KiB\)", combined
    )
    if m:
        metrics["internal_ram_kib"] = m.group(2)

    m = re.search(r"weights \(ro\)\s*:\s*([\d,]+)\s*B\s*\(([\d.]+)\s*KiB\)", combined)
    if m:
        metrics["weights_flash_kib"] = m.group(2)

    m = re.search(r"hyperRAM\s+\[[^\]]+\]:\s+([\d.]+)\s+kB\s*/", combined)
    if m and float(m.group(1)) > 0:
        metrics["external_ram_kib"] = m.group(1)

    # ── Inference time from stedgeai validate output ──
    m = re.search(r"duration\s*:\s*([\d.]+)\s*ms", combined)
    if m:
        metrics["inference_time_ms"] = m.group(1)

    # ── Compute inf/sec from inference time ──
    if metrics["inference_time_ms"]:
        try:
            ms = float(metrics["inference_time_ms"])
            if ms > 0:
                metrics["inf_per_sec"] = f"{1000.0 / ms:.2f}"
        except ValueError:
            pass

    # ── Fallback: parse network_c_info.json directly ──
    if not metrics["internal_ram_kib"] or not metrics["weights_flash_kib"]:
        cinfo_path = _find_network_c_info()
        if cinfo_path:
            _parse_network_c_info(cinfo_path, metrics)

    # ── Fallback: parse Hydra output log ──
    hydra_dir = _find_hydra_output_dir(stdout)
    if hydra_dir:
        _parse_hydra_log(hydra_dir, metrics)
        stm32ai_cinfo = hydra_dir / "stm32ai_files" / "network_c_info.json"
        if stm32ai_cinfo.exists() and (
            not metrics["internal_ram_kib"] or not metrics["weights_flash_kib"]
        ):
            _parse_network_c_info(stm32ai_cinfo, metrics)

    return metrics
