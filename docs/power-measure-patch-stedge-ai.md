# Patching ST Edge AI for power measurement (`avg_power_mW`)

This file is the **canonical** guide for the power-measure ST Edge AI patch (single-file `aiValidation_ATON.c` workflow).

This project benchmarks inference on the STM32N6570-DK using ST Edge AI (`stedgeai`, `n6_loader.py`) and can record **average power during NPU inference** when you use an external INA228 monitor (Arduino sketch in `external/fyp-power-measure/`).

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

1. Flash `external/fyp-power-measure/fyp-power-measure.ino`. It uses **interrupt-driven edge detection** on the sync pin and the **INA228 energy accumulator** for accurate power measurement. Sends binary **protobuf messages** (PowerSample) over serial at 921600 baud. Wire **STM32 sync** (**PD6** by default) to **`IS_INFERENCING_PIN`** (GPIO 3) and common ground.

2. Second serial for the INA228 (not the ST-LINK port used by `stedgeai`):

   - `BENCHMARK_POWER_SERIAL` — e.g. `/dev/ttyUSB1`
   - `BENCHMARK_POWER_BAUD` — default `921600` (matches Arduino sketch)

3. Optional — trim **start/end** of each inference window to reduce GPIO and rail edge effects:

   - `BENCHMARK_POWER_DISCARD_START_MS` — drop first *n* milliseconds of each inference window (default **`1`** ms).
   - `BENCHMARK_POWER_DISCARD_END_MS` — drop last *n* milliseconds of each window (default **`1`** ms).
   - `BENCHMARK_POWER_DISCARD_EDGE_MS` — convenience: sets **both** start and end when the two variables above are **not** set.

   Set **`BENCHMARK_POWER_DISCARD_START_MS=0`** and **`BENCHMARK_POWER_DISCARD_END_MS=0`** to disable trimming.

4. `pip install pyserial protobuf`

5. Run the benchmark. While `BENCHMARK_POWER_SERIAL` is set, a background thread starts at benchmark launch and **appends every protobuf sample** to **`results/benchmark/power-measure.csv`** with a leading **`host_time_iso`** column (UTC). **`avg_power_mW`** in `benchmark_results.csv` uses only samples captured during each model’s **validate** step with `is_inference=True`.

## After an ST Edge AI upgrade

1. Re-paste **`aiValidation_ATON_power_sync.inc.c`** (and `#include "stm32n6xx_hal.h"` + call sites if the vendor file reset).
2. Rebuild and flash.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| Empty `avg_power_mW` | `BENCHMARK_POWER_SERIAL` unset, wrong port, `pyserial`/`protobuf` missing |
| Power looks like whole-run average | Sync GPIO not wired; benchmark falls back to all samples |
| `avg_power_mW` unchanged after discard vars | Edge trimming uses duration-based proportional reduction |
| Want no edge trimming | `BENCHMARK_POWER_DISCARD_START_MS=0` and `BENCHMARK_POWER_DISCARD_END_MS=0` |
| Build errors in the new block | Pin/port conflict or RIF/security; change `PWR_SYNC_*` / RCC macro |
| HAL / GPIO undeclared | `stm32n6xx_hal.h` include path / N6 build only |
| Serial read errors | Check baud rate is 921600, Arduino sketch flashed correctly |
| Protobuf import errors | `pip install protobuf`, regenerate with `protoc --python_out=scripts/benchmark external/fyp-power-measure/power_sample.proto` |
