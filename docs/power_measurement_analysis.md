# Power Measurement Analysis

## Overview

Power measurements are captured using ESP32 + INA228 hardware monitoring the STM32N6570-DK board during benchmarking. This document explains what happens during idle and inference measurements, and how to interpret the results.

## Measurement Architecture

**Hardware:**
- ESP32 + INA228 sensor monitors STM32N6570-DK power consumption
- GPIO pin 3 (IS_INFERENCING_PIN) carries sync signal from STM32
- INA228 configured for continuous measurement at 50μs conversion time
- No averaging filter (COUNT_1) for raw measurements

**Software:**
- `external/fyp-power-measure/fyp-power-measure.ino` - ESP32 firmware
- `src/benchmark/execution/power_serial.py` - Python capture and analysis
- STM32 validation firmware in `/home/fn/ST/STEdgeAI/4.0/Projects/STM32N6570-DK/Applications/NPU_Validation/`

## Measurement Windows

### Idle Window (pm_avg_idle_)

**Trigger:** Sync pin LOW → HIGH (rising edge ends idle window)

**STM32 State During Idle:**
- CPU active, running main loop in `aiValidationProcess()`
- Clock configuration (see Clock Configurations section below)
- NPU powered but idle
- USB/UART communication active
- Caches enabled/disabled per `app_config.h`
- SLEEPDEEP bit cleared (no deep sleep)
- Background tasks: HAL tick, interrupts, external memory refresh

**Measurement:**
- INA228 energy accumulator integrates power continuously
- On rising edge: ESP32 reads accumulated energy and duration
- Sample sent with `is_inference=false`
- Accumulator resets for next window

### Inference Window (pm_avg_inf_)

**Trigger:** Sync pin HIGH → LOW (falling edge ends inference window)

**STM32 State During Inference:**
- **CPU in WFE (Wait-For-Event) sleep mode** - low power state
- NPU actively executing model inference
- Same clock/cache configuration as idle
- USB/UART peripherals still active
- CPU wakes on NPU completion interrupt

**Key Code (`ai_wrapper_ATON_ST_AI.c:1170-1182`):**
```c
rc = stai_network_run(network, STAI_MODE_ASYNC);
do {
    if (rc == STAI_RUNNING_WFE) {
        LL_ATON_OSAL_WFE();  // __WFE() - CPU sleeps here
    }
    rc = stai_ext_network_run_continue(network);
} while ((rc != STAI_DONE) && (rc != STAI_SUCCESS));
```

**Measurement:**
- INA228 accumulator integrates power during NPU execution
- On falling edge: ESP32 reads accumulated energy and duration
- Sample sent with `is_inference=true`
- Accumulator resets for next window

## Power Metrics Interpretation

### pm_avg_idle_mW
- **Baseline system power**
- CPU active + NPU idle + peripherals active
- Averaged across all idle windows in validation run

### pm_avg_inf_mW
- **System power during inference**
- CPU sleeping (WFE) + NPU active + peripherals active
- Averaged across all inference windows

### pm_avg_delta_mW = pm_avg_inf_mW - pm_avg_idle_mW
- **Net system power increase during inference**
- **NOT pure NPU power**
- Formula: `(NPU_active + CPU_sleep) - (NPU_idle + CPU_active)`
- Simplifies to: `NPU_power_increase - CPU_power_decrease`
- **Delta underestimates pure NPU power** because CPU sleep reduces consumption

## Idle Stability Assessment

**pm_avg_idle_ is stable enough for research use:**

✓ **Strengths:**
- Hardware energy accumulator provides accurate integration
- No software averaging filter (raw measurements)
- Edge-triggered windows eliminate timing jitter
- Multiple samples per run enable statistical averaging
- Consistent baseline across all measurements

⚠ **Considerations:**
- Idle includes USB/UART communication overhead
- Background tasks (HAL tick, interrupts) cause minor variations
- Clock configuration significantly affects baseline
- External memory refresh may contribute to variations
- "Idle" is not a pure sleep state - it's active system baseline

## Recommendations for Research

1. **Report all three metrics:** idle, inference, and delta power
2. **Interpret delta correctly:** Net system power increase, not isolated NPU power
3. **Note CPU sleep behavior:** Acknowledge that CPU enters WFE during inference
4. **Maintain consistency:** Keep clock mode and cache settings constant across runs
5. **Use averaged values:** Already implemented via `num_inferences` parameter
6. **Calculate pure NPU power (if needed):**
   ```
   NPU_power ≈ delta + (CPU_active_power - CPU_sleep_power)
   ```
   Requires separate characterization of CPU active vs. sleep power

## Calculation Details

From `src/benchmark/execution/power_serial.py:89-174`:

```python
# Inference metrics
inf_energy_j = sum(energy_j for samples where is_inference=True)
inf_duration_us = sum(duration_us for samples where is_inference=True)
pm_avg_inf_mW = (inf_energy_j / inf_duration_s) * 1000.0

# Idle metrics
idle_energy_j = sum(energy_j for samples where is_inference=False)
idle_duration_us = sum(duration_us for samples where is_inference=False)
pm_avg_idle_mW = (idle_energy_j / idle_duration_s) * 1000.0

# Delta
pm_avg_delta_mW = pm_avg_inf_mW - pm_avg_idle_mW
```

All metrics are averaged over `num_inferences` runs for statistical stability.

## Clock Configurations

The STM32N6570-DK supports multiple clock configurations that significantly affect power consumption. The benchmark patches `app_config.h`: `USE_OVERDRIVE` plus (when `USE_OVERDRIVE` is 0) whether `NO_OVD_CLK400` is defined—see selection logic below.

| Mode | Function | CPU Clock | NPU Clock | PLL Configuration | Usage |
|------|----------|-----------|-----------|-------------------|-------|
| **Overdrive** | `SystemClock_Config_HSI_overdrive()` | 800 MHz | 1000 MHz (1 GHz) | PLL1=800MHz, PLL2=1GHz, PLL3=900MHz | Benchmark `--mode overdrive` (`USE_OVERDRIVE` 1) |
| **No Overdrive (nominal)** | `SystemClock_Config_HSI_no_overdrive()` | 600 MHz | 600 MHz | PLL1=800MHz, PLL2=600MHz | Benchmark `--mode nominal` (`USE_OVERDRIVE` 0, `NO_OVD_CLK400` commented out) |
| **400 MHz (underdrive)** | `SystemClock_Config_HSI_400()` | 400 MHz | 400 MHz | PLL1=800MHz/2 | Benchmark `--mode underdrive` (`USE_OVERDRIVE` 0, `#define NO_OVD_CLK400` active) |
| **64MHz** | `SystemClock_Config_64MHZ()` | 64 MHz | 64 MHz | HSI direct | Not used in benchmarks |

**PLL Details (from `system_clock_config.c`):**

**Overdrive Mode:**
- PLL1: 64MHz × 25 / 2 / 1 / 1 = 800 MHz (CPU via IC1)
- PLL2: 64MHz × 125 / 8 / 1 / 1 = 1000 MHz (NPU via IC6)
- PLL3: 64MHz × 225 / 16 / 1 / 1 = 900 MHz (AXISRAM3-6)

**No Overdrive Mode:**
- PLL1: 64MHz × 25 / 2 / 1 / 1 = 800 MHz (SYSCLK)
- PLL2: 64MHz × 75 / 8 / 1 / 1 = 600 MHz (CPU and NPU)

**400MHz Mode:**
- PLL1: 64MHz × 25 / 2 / 1 / 1 = 800 MHz
- CPU: PLL1 / 2 = 400 MHz (via IC1)
- NPU: PLL1 / 2 = 400 MHz (via IC6)

**Selection Logic (from `main.c:74-84`):**
```c
#if USE_OVERDRIVE
  upscale_vddcore_level();
  SystemClock_Config_HSI_overdrive();
#else
  #ifdef NO_OVD_CLK400
    SystemClock_Config_HSI_400();
  #else
    SystemClock_Config_HSI_no_overdrive();
  #endif
#endif
```

The benchmark workflow patches `app_config.h` for the selected mode (`underdrive`, `nominal`, or `overdrive`), which directly controls CPU and NPU clock frequencies and significantly impacts both idle and inference power measurements.
