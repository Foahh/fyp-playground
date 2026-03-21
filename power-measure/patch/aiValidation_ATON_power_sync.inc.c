/**
 * Paste into ST Edge AI: Middlewares/ST/AI/Validation/Src/aiValidation_ATON.c
 *
 * 1) After the system #includes (e.g. after #include <string.h>), add once if missing:
 *        #include "stm32n6xx_hal.h"
 * 2) Paste this entire file’s body immediately AFTER the _dumpable_tensor_name[] block
 *    (after the closing "};" and before _APP_VERSION_MAJOR_ / struct aton_context).
 * 3) Ensure call sites exist (see docs/power-measure-patch-stedge-ai.md).
 *
 * Or copy the same block from a working aiValidation_ATON.c in this repo’s ST install.
 */

/* -------------------------------------------------------------------------- */
/* External power measurement: GPIO high during npu_run (INA228 sync on host). */
/* Set PWR_MEASUREMENT_SYNC_ENABLE to 0 to disable. Change PWR_SYNC_* for wiring. */
/* -------------------------------------------------------------------------- */
#ifndef PWR_MEASUREMENT_SYNC_ENABLE
#define PWR_MEASUREMENT_SYNC_ENABLE 1
#endif

#if PWR_MEASUREMENT_SYNC_ENABLE

#ifndef PWR_SYNC_GPIO_PORT
#define PWR_SYNC_GPIO_PORT GPIOD
#endif
#ifndef PWR_SYNC_GPIO_PIN
#define PWR_SYNC_GPIO_PIN GPIO_PIN_6
#endif
#ifndef PWR_SYNC_GPIO_RCC_ENABLE
#define PWR_SYNC_GPIO_RCC_ENABLE() __HAL_RCC_GPIOD_CLK_ENABLE()
#endif

static void power_measurement_sync_init(void)
{
  PWR_SYNC_GPIO_RCC_ENABLE();

  GPIO_InitTypeDef gpio = {0};
  gpio.Pin = PWR_SYNC_GPIO_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(PWR_SYNC_GPIO_PORT, &gpio);
  HAL_GPIO_WritePin(PWR_SYNC_GPIO_PORT, PWR_SYNC_GPIO_PIN, GPIO_PIN_RESET);
}

static void power_measurement_sync_begin(void)
{
  HAL_GPIO_WritePin(PWR_SYNC_GPIO_PORT, PWR_SYNC_GPIO_PIN, GPIO_PIN_SET);
}

static void power_measurement_sync_end(void)
{
  HAL_GPIO_WritePin(PWR_SYNC_GPIO_PORT, PWR_SYNC_GPIO_PIN, GPIO_PIN_RESET);
}

#else /* !PWR_MEASUREMENT_SYNC_ENABLE */

static void power_measurement_sync_init(void) {}
static void power_measurement_sync_begin(void) {}
static void power_measurement_sync_end(void) {}

#endif /* PWR_MEASUREMENT_SYNC_ENABLE */
