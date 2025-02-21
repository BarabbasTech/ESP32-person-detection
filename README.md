# ESP32-CAM Person Detection

A TensorFlow Lite Micro implementation for person detection using ESP32-CAM with ESP-IDF framework.

## Features

- Real-time person detection using TensorFlow Lite
- Optimized for ESP32-CAM's limited resources
- LED indication for detected persons
- PSRAM utilization for improved performance

## Hardware Requirements

- ESP32-CAM AI-Thinker board
- USB-UART programmer
- LED (connected to GPIO 4)
- 5V power supply

## Prerequisites

- ESP-IDF v4.1
- Python 3.8 or later
- Git

## Environment Setup

```bash
# Clone ESP-IDF v4.1
git clone -b v4.1 --recursive https://github.com/espressif/esp-idf.git

# Set up environment variables (Windows PowerShell)
$env:IDF_PATH = "C:\Users\<username>\esp\esp-idf"
. $env:IDF_PATH\export.ps1

# For Linux/macOS
export IDF_PATH=~/esp/esp-idf
. $IDF_PATH/export.sh
```

## Building

```bash
# Configure project
idf.py menuconfig

# Build the project
idf.py build

# Flash to ESP32
idf.py -p COMx flash

# Monitor output
idf.py -p COMx monitor
```

## Configuration

Key settings in menuconfig:
1. Set "Camera Configuration" → "Camera Model" → "AI-THINKER"
2. Enable PSRAM support
3. Set CPU frequency to 240MHz
4. Configure flash size to 4MB

## Pin Configuration

| Function | GPIO Pin |
|----------|----------|
| LED      | GPIO 4   |
| Camera XCLK | GPIO 0  |
| Camera PCLK | GPIO 22 |
| Camera VSYNC| GPIO 25 |
| Camera HREF | GPIO 23 |
| Camera SDA  | GPIO 26 |
| Camera SCL  | GPIO 27 |
| Camera D0-D7| GPIO 5,18,19,21,36,39,34,35 |

## Performance

- Detection time: ~5000ms per cycle
- Memory usage: ~64KB tensor arena
- Accuracy: Configurable threshold (default: 200)

## Troubleshooting

1. Memory Issues:
   - Ensure PSRAM is enabled
   - Reduce tensor arena size if needed
   - Check heap fragmentation

2. Camera Issues:
   - Verify pin connections
   - Check power supply stability
   - Ensure camera model selection is correct

## License

Apache License 2.0 - see LICENSE file for details

## Acknowledgments

- TensorFlow Lite Micro team
- Espressif Systems
- AI-Thinker
