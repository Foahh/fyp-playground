"""Output and log parsing helpers for extracting metrics."""

import json
import re
from pathlib import Path


def parse_metrics(stdout: str, stderr: str, cinfo_path: Path | None = None) -> dict:
    """Extract metrics from stedgeai output and network_c_info.json."""
    metrics = {
        "ap_50": "",
        "internal_ram_kib": "",
        "external_ram_kib": "",
        "weights_flash_kib": "",
        "input_buffer_kib": "",
        "output_buffer_kib": "",
        "inference_time_ms": "",
        "inf_per_sec": "",
    }

    combined = stdout + "\n" + stderr

    # ── Read network_c_info.json for precise memory values ──
    if cinfo_path is not None and cinfo_path.exists():
        try:
            with open(cinfo_path, "r") as f:
                cinfo = json.load(f)

            # Internal memory pools (on-chip)
            INTERNAL_POOLS = {
                "flexMEM",
                "cpuRAM1",
                "cpuRAM2",
                "npuRAM3",
                "npuRAM4",
                "npuRAM5",
                "npuRAM6",
                "npuCache",
                "vencRAM",
            }
            # External memory pools (off-chip)
            EXTERNAL_POOLS = {"hyperRAM", "octoFlash", "xSPI3", "sdRAM"}

            pools = cinfo.get("memory_pools", [])
            internal_total = 0
            external_total = 0

            for pool in pools:
                name = pool.get("name", "")
                # Skip merged pools (have subpools)
                if pool.get("subpools"):
                    continue
                used = pool.get("used_size_bytes", 0)
                if name in INTERNAL_POOLS and "WRITE" in pool.get("rights", ""):
                    internal_total += used
                elif name in EXTERNAL_POOLS and "WRITE" in pool.get("rights", ""):
                    external_total += used

            if internal_total >= 0:
                metrics["internal_ram_kib"] = f"{internal_total / 1024:.2f}"
            if external_total >= 0:
                metrics["external_ram_kib"] = f"{external_total / 1024:.2f}"

            # Weights from memory_footprint
            mem = cinfo.get("memory_footprint", {})
            if mem.get("weights", 0) >= 0:
                metrics["weights_flash_kib"] = f"{mem.get('weights', 0) / 1024:.2f}"

            # I/O buffer sizes from graphs
            graphs = cinfo.get("graphs", [])
            if graphs:
                g0 = graphs[0]
                buffers = {b["id"]: b["size_bytes"] for b in cinfo.get("buffers", [])}

                inp_total = sum(buffers.get(i, 0) for i in g0.get("inputs", []))
                out_total = sum(buffers.get(i, 0) for i in g0.get("outputs", []))

                if inp_total >= 0:
                    metrics["input_buffer_kib"] = f"{inp_total / 1024:.2f}"
                if out_total >= 0:
                    metrics["output_buffer_kib"] = f"{out_total / 1024:.2f}"
        except Exception:
            pass

    # ── AP@50 ──
    m = re.search(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s+\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([\d.]+)",
        combined,
    )
    if m:
        metrics["ap_50"] = f"{float(m.group(1)) * 100:.2f}"

    if not metrics["ap_50"]:
        m = re.search(r"Mean AP \(mAP\):\s+([\d.]+)", combined)
        if m:
            metrics["ap_50"] = m.group(1)

    if not metrics["ap_50"]:
        m = re.search(r"AP50\s*:\s*([\d.]+)", combined)
        if m:
            val = float(m.group(1))
            metrics["ap_50"] = f"{val * 100:.2f}" if val <= 1.0 else f"{val:.2f}"

    # ── Inference time ──
    m = re.search(r"Inference time: avg=([\d.]+)ms", combined)
    if m:
        metrics["inference_time_ms"] = m.group(1)

    if not metrics["inference_time_ms"]:
        m = re.search(r"duration\s*:\s*([\d.]+)\s*ms", combined)
        if m:
            metrics["inference_time_ms"] = m.group(1)

    # ── Compute inf/sec ──
    if metrics["inference_time_ms"]:
        try:
            ms = float(metrics["inference_time_ms"])
            if ms > 0:
                metrics["inf_per_sec"] = f"{1000.0 / ms:.2f}"
        except ValueError:
            pass

    return metrics
