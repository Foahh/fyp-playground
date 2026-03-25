#include <Adafruit_INA228.h>
#include <Wire.h>

Adafruit_INA228 ina228;

static constexpr uint8_t INA228_I2C_ADDR = INA228_I2CADDR_DEFAULT;
static constexpr float SHUNT_RESISTOR_OHMS = 0.1f; // R100
static constexpr float MAX_EXPECTED_CURRENT_A = 10.0f;

static constexpr int INA228_SDA_PIN = 2;
static constexpr int INA228_SCL_PIN = 1;
static constexpr int INA228_ALERT_PIN = 0;

static constexpr int IS_INFERENCING_PIN = 3;

// Min SAMPLE_INTERVAL_US ~= INA228_COUNT_X * (t_shunt + t_bus) in us.
static constexpr uint32_t SAMPLE_INTERVAL_US = 20000;

static uint32_t next_sample_us = 0;
static char line_buf[128];

static void waitForStartCommand() {
  Serial.println("# Ready. Send START to begin sampling.");
  while (true) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.length() > 0 && line.equalsIgnoreCase("START")) {
        Serial.println("# Streaming started.");
        return;
      }
    }
    delay(10);
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(INA228_SDA_PIN, INA228_SCL_PIN);
  Wire.setClock(400000);
  pinMode(INA228_ALERT_PIN, INPUT_PULLUP);
  pinMode(IS_INFERENCING_PIN, INPUT_PULLDOWN);

  Serial.println("# INA228 monitor starting...");
  while (true) {
    if (!ina228.begin(INA228_I2C_ADDR, &Wire)) {
      Serial.println("ERROR: INA228 not found.");
    } else {
      break;
    }
    delay(1000);
  }

  ina228.setShunt(SHUNT_RESISTOR_OHMS, MAX_EXPECTED_CURRENT_A);

  ina228.setAveragingCount(INA228_COUNT_16);
  ina228.setVoltageConversionTime(INA228_TIME_540_us);
  ina228.setCurrentConversionTime(INA228_TIME_540_us);

  ina228.setAlertType(INA228_ALERT_CONVERSION_READY);
  ina228.setAlertPolarity(INA228_ALERT_POLARITY_INVERTED);
  ina228.resetAccumulators();

  waitForStartCommand();

  Serial.println("ts_us,current_mA,bus_V,power_mW,inference");
  next_sample_us = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - next_sample_us) < 0) {
    return;
  }
  next_sample_us += SAMPLE_INTERVAL_US;

  if (!ina228.conversionReady()) {
    return;
  }

  const uint32_t ts = micros();
  const float current_mA = ina228.getCurrent_mA();
  const float bus_V = ina228.getBusVoltage_V();
  const float power_mW = ina228.getPower_mW();
  const int inference = digitalRead(IS_INFERENCING_PIN) == HIGH ? 1 : 0;

  int len = snprintf(line_buf, sizeof(line_buf), "%lu,%.3f,%.4f,%.3f,%d\n",
                     (unsigned long)ts, current_mA, bus_V, power_mW, inference);
  Serial.write(line_buf, len);
}
