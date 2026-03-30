## Role

You are an expert in embedded ML and edge AI, specialising in model selection for resource-constrained wearable MCUs.

---

## Hardware & Application Summary

**Board:** STM32N6570-DK (Cortex-M55 + ATON NPU, ST Edge AI / `ll_aton` runtime)  
**Application:** Smart glass prototype — real-time hazard detection + hand-proximity alerts (wearable safety device)  
**Camera:** IMX335 → DCMIPP (PIPE2 feeds NN, PIPE1 feeds display)  
**Depth sensor:** VL53L5CX 8×8 ToF, fused with detections for 3D proximity  
**RTOS:** ThreadX  
**Output:** Bounding boxes + confidence scores → ToF fusion + host protocol  

All candidates will be **fine-tuned on a custom dataset** after selection. Pre-trained `ap_50` is an architecture capacity proxy, not deployment accuracy.

### Pre-Verified (Not Differentiating)

- All benchmarked models are ST Edge AI compatible.
- Post-processing overhead is ~200 ns — negligible.
- All models are Int8 or W4A8 (both NPU-compatible).

---

## Memory Map & Constraints

| Pool | Capacity | Type | Role |
|---|---|---|---|
| cpuRAM2 (AXISRAM2) | 512 KiB | On-chip | CPU+NPU shared activations |
| npuRAM3–6 (AXISRAM3–6) | 4 × 448 KiB | On-chip | NPU-local activations |
| hyperRAM (xSPI1) | 16 MiB | Off-chip | Activation spillover |
| octoFlash (xSPI2) | 60 MiB | Off-chip | Weights (read-only) |

**Total on-chip activation capacity:** 512 + 4 × 448 = **2304 KiB**

### Hard Constraints

| Constraint | Limit | Column |
|---|---|---|
| Latency | ≤ 66.67 ms (= 1000/15 FPS) | `inference_time_ms` |
| Weights flash | ≤ 61440 KiB | `weights_flash_kib` |

### Derived Memory Metrics

The benchmark includes I/O buffers in activation totals, but the **firmware** places them outside the activation pools (input → PSRAM, output → cpuRAM1). Correct for this:

```
activations_without_io = (internal_ram_kib + external_ram_kib) - (input_buffer_kib + output_buffer_kib)
estimated_spill_kib    = max(0, activations_without_io - 2304)
```

Models with `estimated_spill_kib > 0` are flagged `[ExtRAM]` but **not excluded** — HyperRAM spill is penalised via scoring, not filtered. Memory is a soft penalty, not a hard constraint.

### Reference Deployment Anchor

`st_yolo_x_nano_480` (480×480): ~2195.5 KiB on-chip + ~450 KiB HyperRAM spill → **44 ms inference**. Proves HyperRAM spillover is viable with acceptable latency impact.

---

## Benchmark Data

### Column Definitions

| Column | Description |
|---|---|
| `model_family` | Architecture family |
| `model_variant` | Specific variant |
| `hyperparameters` | Scaling parameters (if any) |
| `dataset` | `COCO-80` (80 classes) or `COCO-Person` (1 class) |
| `format` | `Int8` or `W4A8` |
| `resolution` | Input resolution (square) |
| `internal_ram_kib` | Total on-chip activation memory (all internal pools) |
| `external_ram_kib` | HyperRAM spillover (0 = fully on-chip) |
| `weights_flash_kib` | Weights in octoFlash |
| `input_buffer_kib` | Input buffer (already counted in RAM totals) |
| `output_buffer_kib` | Output buffer (already counted in RAM totals) |
| `inference_time_ms` | Inference latency |
| `inf_per_sec` | Throughput |
| `ap_50` | Average Precision @ IoU 0.50 |
| `pm_avg_inf_mW` | Avg power during inference (mW) |
| `pm_avg_idle_mW` | Avg idle power (mW) |
| `pm_avg_delta_mW` | Inference power overhead (mW) |
| `pm_avg_inf_ms` | Inference window duration (ms) |
| `pm_avg_idle_ms` | Idle window duration (ms) |
| `pm_avg_inf_mJ` | Energy per inference (mJ) |
| `pm_avg_idle_mJ` | Idle energy (mJ) |

**Note:** `internal_ram_kib + external_ram_kib` = total activations (already includes I/O buffers). Do not add `input_buffer_kib`/`output_buffer_kib` on top.

### Raw Data

```csv
[CSV_DATA]
```

---

## Pre-Scoring Issues (resolve in order, document in Section 1)

### Issue 1 — Dataset Incomparability

`COCO-Person` and `COCO-80` AP scores are not comparable. Maintain separate columns `ap_50_person` and `ap_50_coco80`. Use `ap_50_person` as primary when available. For COCO-80-only models, normalise within the COCO-80 cohort separately and flag as `[80only]` with note: *"COCO-80 only — conservative accuracy signal; fine-tuning uplift may be greater."*

### Issue 2 — Fine-Tuning Context

Weight accuracy-efficiency ratios (`ap_50 / inference_time_ms`, `ap_50 / pm_avg_inf_mJ`) more heavily than raw `ap_50`. Fine-tuning closes accuracy gaps but cannot recover inference budget. Don't penalise lower absolute `ap_50` if the efficiency curve is strong.

### Issue 3 — Resolution Variants

Don't auto-select highest resolution. After hard constraint filtering, pick the resolution with best composite score as family representative. If two resolutions score within 5%, list both as viable operating points with trade-off notes.

### Issue 4 — Quantisation Format Selection

For families with both Int8 and W4A8 at the same resolution: compare across all metrics, state rationale. **Prefer Int8** for fine-tuning stability unless W4A8 is the only format passing flash constraint or offers decisive advantage on multiple efficiency dimensions simultaneously.

### Issue 5 — HyperRAM Spillover

Compute `activations_without_io` and `estimated_spill_kib` for every row. Flag `[ExtRAM]` variants but keep them in ranking. Spill is penalised via the memory scoring term and implicitly via latency/energy terms.

### Issue 6 — Buffer Placement Correction

Use `activations_without_io` (not raw RAM columns) for the memory scoring term. Note that models with large `input_buffer_kib` benefit most from firmware PSRAM placement.

---

## Scoring Framework

### Constraint Gate

**Remove** rows violating hard constraints before scoring. Never score a disqualified row.

### Criteria

| Dimension | Tier | Metric | Direction |
|---|---|---|---|
| Real-time latency | 🔴 Hard | `inference_time_ms` ≤ 66.67 ms | Gate |
| Weights flash | 🔴 Hard | `weights_flash_kib` ≤ 61440 KiB | Gate |
| Architecture capacity | 🟡 Primary | `ap_50` (within-cohort normalised) | Higher better |
| Energy per inference | 🟡 Primary | `pm_avg_inf_mJ` | Lower better |
| Efficiency curve | 🟡 Primary | `ap_50 / inference_time_ms` (and/or geometric mean with `ap_50 / pm_avg_inf_mJ` if power data available) | Higher better |
| Activation memory footprint | 🟡 Primary | `activations_without_io` (derived) | Lower better |
| Latency margin | 🟢 Secondary | `inference_time_ms` (among passing candidates) | Lower better |
| Architecture modernity | 🟢 Secondary | 1.0 = anchor-free decoupled head; 0.5 = anchor-based + modern backbone; 0.0 = legacy anchor-based | Higher better |

Assign explicit weights summing to **1.0**. Justify each weight. If power columns are missing, state this and omit/down-weight energy terms.

### Composite Score Formula (Option A — recommended)

```
score = w_acc    * norm_cohort(ap_50)
      + w_energy * norm(1 / pm_avg_inf_mJ)
      + w_eff    * norm(efficiency_metric)
      + w_lat    * norm(1 / inference_time_ms)
      + w_mem    * norm(1 / activations_without_io)
      + w_modern * architecture_modernity_score
```

`norm(x)` = min-max across constraint-passing candidates. `norm_cohort(ap_50)` = min-max within each dataset cohort separately.

**Option B** (if explicit spill emphasis desired): split `w_mem` into `w_head * norm(1/activations_without_io) + w_spill * norm(1/(1 + estimated_spill_kib))`.

State which option and final weights before presenting scores.

---

## Output Format

### Section 0 — Assumptions & Substituted Values

List any unfilled `[PLACEHOLDER]` values with assumed values. If benchmark is incomplete, list missing columns/rows and impact on scoring.

### Section 1 — Data Cleaning & Anomaly Report

Document decisions for each Issue (1–6). Close with disposition table:

| Model Variant | Format | Res | internal_ram | external_ram | input_buf | output_buf | activ_no_io | est_spill | Flags | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|

### Section 2 — Criteria Framework Table

| Criterion | Tier | Metric | Weight | Rationale |
|---|---|---|---|---|

Confirm weights sum to 1.0.

### Section 3 — Composite Scoring & Ranked Table

Restate formula with weights, then:

| Rank | Model Variant | Format | Res | Dataset | acc | energy | delta | eff | lat | mem | modern | **Score** | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

### Section 4 — Interpretation

For each ranked candidate (cover as many as needed for a clear recommendation):

1. **Configuration** — family, variant, format, resolution
2. **Score drivers** — which components drove its rank
3. **Memory profile** — `activations_without_io` vs 2304 KiB, spill vs reference anchor
4. **Caveats** — spill magnitude, COCO-80-only, resolution trade-offs, missing data
5. **Recommended operating point** — resolution + format for fine-tuning, with reasoning
6. **Fine-tuning suitability** — head design, anchor strategy, backbone depth vs dataset size

### Power Measurement Methodology

All power columns (`pm_avg_*`) are captured via an **ESP32 + INA228** hardware monitor on the STM32N6570-DK power rail, synchronised to inference via a GPIO edge signal. The INA228 hardware energy accumulator integrates continuously at 50 μs conversion time with no averaging filter — edge-triggered windows eliminate software timing jitter. Metrics are averaged over all inference/idle windows in a run.

**What each window captures:**

| Window | CPU state | NPU state | Peripherals |
|---|---|---|---|
| **Idle** (`pm_avg_idle_*`) | Active (main loop) | Powered but idle | USB/UART active |
| **Inference** (`pm_avg_inf_*`) | **WFE sleep** (wakes on NPU interrupt) | Actively executing model | USB/UART active |

**Interpreting `pm_avg_delta_mW`:** This is `pm_avg_inf_mW − pm_avg_idle_mW`, i.e. the net *system-level* power change during inference. It is **not** pure NPU power — it reflects `(NPU active + CPU sleep) − (NPU idle + CPU active)`, so the CPU's transition to WFE sleep partially offsets the NPU's power draw. Delta **underestimates** isolated NPU power. For model comparison this is acceptable: the offset is approximately constant across models, so delta remains a valid **relative** ranking signal.

**`pm_avg_inf_mJ`** (energy per inference) is the most battery-relevant metric: it integrates power over the full inference duration, so faster models naturally benefit even at equal power draw.

**Clock configuration:** All benchmarks use the same clock mode (underdrive, nominal, or overdrive, as indicated by the dataset). Results across different clock modes are not directly comparable.