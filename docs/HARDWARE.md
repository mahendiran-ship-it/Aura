# Hardware integration

AURA's runtime contains **no simulated sensors**. A sensor object exists only if real hardware is behind it. Tests use fake serial ports to check parsing; **nothing here has been validated against physical hardware yet.**

## Interfaces (`aura/sensors/base.py`)

- `DistanceSensor.read() -> DistanceReading | None` (ultrasonic, ToF)
- `HapticOutput.pulse(level, direction)` (vibration motors; **interface only, no implementation yet**)
- `SensorHub` polls sensors, ignores readings older than 1 s and sensors that raise.

Readings are turned into alerts by `HazardDetector.from_distance_sensor` using the thresholds in `config/safety.yaml` (`sensor_distance_cm`).

## ESP32 / Arduino distance sensor over serial

Wiring example: HC-SR04 (5 V, use a level shifter or voltage divider on ECHO for 3.3 V boards) or a VL53L1X ToF module (I²C). Have the board print one line per measurement at ~10 Hz:

```
DIST:ahead:123.4        # centimetres; direction = ahead | left | right | down
```

Arduino-style sketch for an HC-SR04 (trigger pin 5, echo pin 18):

```cpp
void setup() { Serial.begin(115200); pinMode(5, OUTPUT); pinMode(18, INPUT); }
void loop() {
  digitalWrite(5, LOW); delayMicroseconds(2);
  digitalWrite(5, HIGH); delayMicroseconds(10); digitalWrite(5, LOW);
  long us = pulseIn(18, HIGH, 30000);
  if (us > 0) { Serial.print("DIST:ahead:"); Serial.println(us * 0.0343 / 2.0); }
  delay(100);
}
```

Host side (`pip install pyserial`):

```python
from aura.core.orchestrator import Orchestrator
from aura.sensors.base import SensorHub
from aura.sensors.serial_sensor import SerialDistanceSensor

hub = SensorHub([SerialDistanceSensor("/dev/ttyUSB0")])
orch = Orchestrator(sensors=hub)
orch.start()
```

Lines that are malformed, outside 1–800 cm, or from unknown directions are ignored. Ultrasonic sensors have blind spots, narrow cones, and fail on soft or angled surfaces — a sensor reading is another *aid*, not a guarantee.

## Cameras

Any source OpenCV can open (USB webcam, UVC wearable camera, RTSP/HTTP stream). For smart glasses or Bluetooth cameras that expose a stream or file, set `camera.source`. For devices with only a vendor SDK, implement `CameraSource` (three methods: `open`, `read`, `close`).

## Audio

Bone-conduction or Bluetooth headsets work as ordinary system audio output for TTS (pair at OS level). Keep ears open to the environment; do not use in-ear isolation outdoors.

## Not implemented

IMU/camera-motion compensation, vibration motor driver, battery telemetry from custom hardware (system/Termux battery is shown only on startup), wearable enclosures, power management.
