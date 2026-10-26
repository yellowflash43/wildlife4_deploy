# Wildlife Detection on Raspberry Pi 5

Real-time wildlife detection using an ONNX model and a Raspberry Pi camera.

## Features

- ONNX-based wildlife inference
- Live camera detection
- Browser-based live video interface
- Console-based detection script

## Project Files

- `wildlife4_v4_demo.onnx` — trained ONNX model
- `wildlife_live.py` — console-based detection
- `wildlife_visual.py` — visual detection interface
- `wildlife_web.py` — browser-based live detection
- `.gitignore` — excludes local and runtime files

## Hardware

- Raspberry Pi 5
- Raspberry Pi camera (IMX708)

## Software Requirements

- Python 3
- ONNX Runtime
- OpenCV
- NumPy
- Other dependencies required by the selected script

## Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

Install the Python dependencies required by the selected script.

## Run

Console detector:

    python wildlife_live.py

Web interface:

    python wildlife_web.py

Open the configured web-server address in a browser on the same network.

## Notes

- Camera access and system packages may require Raspberry Pi-specific configuration.
- Keep virtual environments, logs, process IDs, credentials, and private configuration files out of version control.
- Verify model and dataset licensing before redistributing the trained model.
