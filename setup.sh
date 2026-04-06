#!/usr/bin/env bash
set -e

echo "=== Checking Python version ==="
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "ERROR: Python 3.10 required, found $PYTHON_VERSION"
    exit 1
fi

echo "=== Checking ffmpeg ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Pre-downloading RVC pretrained models ==="
python -c "
from rvc_python.infer import RVCInference
rvc = RVCInference(device='cpu')
print('RVC models ready.')
"

echo "=== Setup complete ==="
