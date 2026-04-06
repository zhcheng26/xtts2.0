#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Checking ffmpeg ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

echo "=== Pre-downloading RVC pretrained models ==="
python -c "
from rvc_python.infer import RVCInference
rvc = RVCInference(device='cpu')
print('RVC models ready.')
"

echo "=== Setup complete ==="
