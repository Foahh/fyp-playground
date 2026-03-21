# Patching ST Edge AI for power measurement (`avg_power_mW`)

This file is the **canonical** guide for the power-measure ST Edge AI patch (single-file `aiValidation_ATON.c` workflow).

This project benchmarks inference on the STM32N6570-DK using ST Edge AI (`stedgeai`, `n6_loader.py`) and can record **average power during NPU inference** when you use an external INA228 monitor (Arduino sketch in `power-measure/`).

After you **upgrade ST Edge AI** or **reinstall** the tree, re-apply the patch below.

## What the patch does

1. **`aiValidation_ATON.c`** — Adds **static** GPIO helpers and calls them around **`npu_run()`** so only **inference** is marked high on a pin (default **PD6** on STM32N6570-DK; checked against BSP for conflicts), not UART/USB protocol I/O.
2. **Host** — Benchmark reads the INA228 serial stream during `validate` when `BENCHMARK_POWER_SERIAL` is set (`scripts/benchmark/`).

There is **no** separate `power_measure_sync.c` and **no** NPU_Validation **Makefile** change: everything lives in one middleware file.

## One-file patch (recommended)

**Source in this repo:** `power-measure/patch/aiValidation_ATON_power_sync.inc.c`

**Target in ST install:**

`$STEDGEAI_CORE_DIR/Middlewares/ST/AI/Validation/Src/aiValidation_ATON.c`

### Steps

1. Open `aiValidation_ATON.c`.
2. With the other `#include` lines, add **once** (if missing):

   ```c
   #include "stm32n6xx_hal.h"
   ```

3. Paste the **code block** from `aiValidation_ATON_power_sync.inc.c` (everything after the file comment) **immediately after** the `_dumpable_tensor_name[]` array (after the closing `};`), **before** `_APP_VERSION_MAJOR_` / `struct aton_context`.

4. **Call sites** — If your file is stock and does not already call the helpers, add:

   - In **`aiValidationInit()`**, after `cyclesCounterInit();`:

     ```c
       power_measurement_sync_init();
     ```

   - In **`aiPbCmdNNRun`**, wrap **`npu_run`**:

     ```c
       power_measurement_sync_begin();
       npu_run(&ctx->instance, &counters);
       power_measurement_sync_end();
     ```

   If ST refactors the file, search for `npu_run(` and keep **begin/end** directly around that call.

5. Rebuild and flash **NPU_Validation** (e.g. `BUILD_CONF=N6-DK` via `n6_loader.py` / benchmark load step).

### Disable without removing code

Define **`PWR_MEASUREMENT_SYNC_ENABLE`** to **`0`** before the pasted block (e.g. in `aiValidation_ATON.c` or project `CFLAGS`), or edit the `#ifndef PWR_MEASUREMENT_SYNC_ENABLE` default in the pasted section.

### Change GPIO

Edit the **`PWR_SYNC_GPIO_*`** / **`PWR_SYNC_GPIO_RCC_ENABLE`** defaults in the pasted block so the RCC macro matches the port (e.g. `__HAL_RCC_GPIOH_CLK_ENABLE()` if you move to port H).

## Arduino and host benchmark

1. Flash `power-measure/power-measure.ino`. It **waits for a `START` command** on serial before streaming CSV (status lines are prefixed with `#`). The benchmark sends `START\\n` automatically when it opens `BENCHMARK_POWER_SERIAL`. For manual use, open the serial console and send `START`. Wire **STM32 sync** (**PD6** by default) to **`SYNC_FROM_MCU_PIN`** and common ground.
2. Second serial for the INA228 (not the ST-LINK port used by `stedgeai`):

   - `BENCHMARK_POWER_SERIAL` — e.g. `/dev/ttyUSB1`
   - `BENCHMARK_POWER_BAUD` — default `115200`

3. Optional — trim **start/end** of each inference window (by CSV `ts_us`) to reduce GPIO and rail edge effects:

   - `BENCHMARK_POWER_DISCARD_START_MS` — drop samples in the first *n* milliseconds of each contiguous `sync==1` segment (default **`1`** ms when unset).
   - `BENCHMARK_POWER_DISCARD_END_MS` — drop samples in the last *n* milliseconds of each segment (default **`1`** ms when unset).
   - `BENCHMARK_POWER_DISCARD_EDGE_MS` — convenience: sets **both** start and end to the same value when the two variables above are **not** set (e.g. `0.5` for 0.5 ms each side).

   Set **`BENCHMARK_POWER_DISCARD_START_MS=0`** and **`BENCHMARK_POWER_DISCARD_END_MS=0`** to disable trimming. If a segment is shorter than the requested trim, or timestamps are missing, the benchmark keeps the full segment for that run (no worse than no discard).

4. `pip install pyserial`
5. Run the benchmark. While `BENCHMARK_POWER_SERIAL` is set, a background thread starts at benchmark launch and **appends every INA228 row** to **`results/benchmark/power-measure.csv`** with a leading **`host_time_iso`** column (UTC). **`avg_power_mW`** in `benchmark_results.csv` still uses only samples captured during each model’s **validate** step (same logic as before, including optional edge discard).

## After an ST Edge AI upgrade

1. Re-paste **`aiValidation_ATON_power_sync.inc.c`** (and `#include "stm32n6xx_hal.h"` + call sites if the vendor file reset).
2. Rebuild and flash.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| Empty `avg_power_mW` | `BENCHMARK_POWER_SERIAL` unset, wrong port, or `pyserial` missing |
| Power looks like whole-run average | Sync GPIO not wired; benchmark falls back to all INA228 lines |
| `avg_power_mW` unchanged after discard vars | `ts_us` missing in CSV; discard needs timestamps |
| Want no edge trimming | `BENCHMARK_POWER_DISCARD_START_MS=0` and `BENCHMARK_POWER_DISCARD_END_MS=0` |
| Build errors in the new block | Pin/port conflict or RIF/security; change `PWR_SYNC_*` / RCC macro |
| HAL / GPIO undeclared | `stm32n6xx_hal.h` include path / N6 build only (this path is for STM32N6 NPU validation) |
