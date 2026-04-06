# Voice Clone + Singing Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio demo where users fine-tune an RVC v2 voice model from audio samples, then use it to convert any Chinese song's vocals into the target person's voice.

**Architecture:** Demucs separates song vocals from accompaniment → RMVPE extracts pitch → RVC v2 (fine-tuned) converts vocals to target voice → pydub mixes converted vocals back with accompaniment. Two Gradio tabs: Tab 1 for training, Tab 2 for inference.

**Tech Stack:** Python 3.10, PyTorch 2.x + CUDA 11.8, demucs>=4.0, infer-rvc-python>=0.2.4, pydub, Gradio 4.x, ffmpeg, pytest

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Python dependencies |
| `setup.sh` | One-time environment setup and model pre-download |
| `modules/__init__.py` | Empty package marker |
| `modules/separator.py` | Demucs vocal/accompaniment separation |
| `modules/mixer.py` | pydub audio mixing with volume control |
| `modules/converter.py` | RVC v2 inference wrapper (VoiceConverter class) |
| `train.py` | Training pipeline: preprocess → extract features → train RVC |
| `infer.py` | Full inference pipeline: separate → convert → mix |
| `app.py` | Gradio UI (Tab 1: Train, Tab 2: Infer) |
| `tests/__init__.py` | Empty package marker |
| `tests/test_separator.py` | Unit tests for separator module |
| `tests/test_mixer.py` | Unit tests for mixer module |
| `tests/test_converter.py` | Unit tests for converter module (mocked RVC) |
| `tests/test_pipelines.py` | Unit tests for train/infer pipelines (mocked) |
| `tests/test_smoke.py` | Interface consistency smoke tests |

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `setup.sh`
- Create: `modules/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p modules tests models uploads outputs
```

- [ ] **Step 2: Write requirements.txt**

```
torch>=2.0.0
torchaudio>=2.0.0
demucs>=4.0.0
gradio>=4.44.0
pydub>=0.25.1
numpy>=1.24.0
scipy>=1.11.0
librosa>=0.10.0
soundfile>=0.12.1
infer-rvc-python>=0.2.4
pytest>=7.4.0
```

- [ ] **Step 3: Write setup.sh**

```bash
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
```

Make it executable:
```bash
chmod +x setup.sh
```

- [ ] **Step 4: Create empty init files**

Create `modules/__init__.py` with empty content.
Create `tests/__init__.py` with empty content.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt setup.sh modules/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding and dependencies"
```

---

### Task 2: Vocal separation module

**Files:**
- Create: `modules/separator.py`
- Create: `tests/test_separator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_separator.py`:
```python
import numpy as np
import soundfile as sf
import os
import pytest
from modules.separator import separate_vocals


@pytest.fixture
def sine_wav(tmp_path):
    """Generate a 5-second 440Hz sine wave as test audio."""
    sr = 44100
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    path = str(tmp_path / "test_song.wav")
    sf.write(path, audio, sr)
    return path


def test_separate_vocals_returns_two_files(sine_wav, tmp_path):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    vocals_path, accompaniment_path = separate_vocals(sine_wav, out_dir)
    assert os.path.exists(vocals_path)
    assert os.path.exists(accompaniment_path)


def test_separate_vocals_output_names(sine_wav, tmp_path):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    vocals_path, accompaniment_path = separate_vocals(sine_wav, out_dir)
    assert vocals_path.endswith("vocals.wav")
    assert accompaniment_path.endswith("accompaniment.wav")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_separator.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'modules.separator'`

- [ ] **Step 3: Implement separator.py**

`modules/separator.py`:
```python
import os
from pathlib import Path

import torch
from demucs.api import Separator, save_audio


def separate_vocals(input_path: str, output_dir: str) -> tuple[str, str]:
    """
    Separate vocals from accompaniment using Demucs htdemucs_ft.

    Args:
        input_path: Path to input audio file (MP3/WAV).
        output_dir: Directory to write vocals.wav and accompaniment.wav.

    Returns:
        (vocals_path, accompaniment_path)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator(model="htdemucs_ft", device=device)

    _, separated = separator.separate_audio_file(Path(input_path))

    vocals_path = os.path.join(output_dir, "vocals.wav")
    accompaniment_path = os.path.join(output_dir, "accompaniment.wav")

    save_audio(separated["vocals"], vocals_path, samplerate=separator.samplerate)

    accompaniment = sum(
        tensor for stem, tensor in separated.items() if stem != "vocals"
    )
    save_audio(accompaniment, accompaniment_path, samplerate=separator.samplerate)

    return vocals_path, accompaniment_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_separator.py -v
```
Expected: `2 passed`

*Note: First run downloads the htdemucs_ft model (~200MB). May take a few minutes.*

- [ ] **Step 5: Commit**

```bash
git add modules/separator.py tests/test_separator.py
git commit -m "feat: add Demucs vocal separation module"
```

---

### Task 3: Audio mixing module

**Files:**
- Create: `modules/mixer.py`
- Create: `tests/test_mixer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_mixer.py`:
```python
import numpy as np
import soundfile as sf
import os
import pytest
from modules.mixer import mix_audio


@pytest.fixture
def two_wav_files(tmp_path):
    sr = 44100
    duration = 3
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    vocals = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    accompaniment = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
    v_path = str(tmp_path / "vocals.wav")
    a_path = str(tmp_path / "accompaniment.wav")
    sf.write(v_path, vocals, sr)
    sf.write(a_path, accompaniment, sr)
    return v_path, a_path


def test_mix_creates_output_file(two_wav_files, tmp_path):
    v_path, a_path = two_wav_files
    output_path = str(tmp_path / "mixed.wav")
    result = mix_audio(v_path, a_path, output_path)
    assert os.path.exists(result)


def test_mix_returns_output_path(two_wav_files, tmp_path):
    v_path, a_path = two_wav_files
    output_path = str(tmp_path / "mixed.wav")
    result = mix_audio(v_path, a_path, output_path)
    assert result == output_path


def test_mix_with_volume_adjustment(two_wav_files, tmp_path):
    v_path, a_path = two_wav_files
    output_path = str(tmp_path / "mixed_vol.wav")
    result = mix_audio(v_path, a_path, output_path, vocals_volume=1.5, accompaniment_volume=0.8)
    assert os.path.exists(result)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mixer.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'modules.mixer'`

- [ ] **Step 3: Implement mixer.py**

`modules/mixer.py`:
```python
import math
from pydub import AudioSegment


def mix_audio(
    vocals_path: str,
    accompaniment_path: str,
    output_path: str,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
) -> str:
    """
    Mix vocals and accompaniment into a single output file.

    Args:
        vocals_path: Path to vocals WAV file.
        accompaniment_path: Path to accompaniment WAV file.
        output_path: Path to write the mixed output WAV file.
        vocals_volume: Volume multiplier for vocals (1.0 = unchanged).
        accompaniment_volume: Volume multiplier for accompaniment (1.0 = unchanged).

    Returns:
        output_path
    """
    vocals = AudioSegment.from_file(vocals_path)
    accompaniment = AudioSegment.from_file(accompaniment_path)

    if vocals_volume != 1.0:
        db_change = 20 * math.log10(max(vocals_volume, 1e-6))
        vocals = vocals + db_change

    if accompaniment_volume != 1.0:
        db_change = 20 * math.log10(max(accompaniment_volume, 1e-6))
        accompaniment = accompaniment + db_change

    mixed = accompaniment.overlay(vocals)
    mixed.export(output_path, format="wav")
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mixer.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add modules/mixer.py tests/test_mixer.py
git commit -m "feat: add pydub audio mixing module"
```

---

### Task 4: RVC voice conversion module

**Files:**
- Create: `modules/converter.py`
- Create: `tests/test_converter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_converter.py`:
```python
import os
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch, MagicMock
from modules.converter import VoiceConverter


@pytest.fixture
def vocals_wav(tmp_path):
    sr = 40000
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
    path = str(tmp_path / "vocals.wav")
    sf.write(path, audio, sr)
    return path


def test_converter_init_accepts_device():
    with patch("modules.converter.RVCInference"):
        vc = VoiceConverter(device="cpu")
    assert vc is not None


def test_load_model_calls_rvc_load(tmp_path):
    model_path = str(tmp_path / "model.pth")
    index_path = str(tmp_path / "model.index")
    open(model_path, "w").close()
    open(index_path, "w").close()

    with patch("modules.converter.RVCInference") as mock_rvc_cls:
        mock_rvc = MagicMock()
        mock_rvc_cls.return_value = mock_rvc
        vc = VoiceConverter(device="cpu")
        vc.load_model(model_path, index_path)
        mock_rvc.load_model.assert_called_once_with(model_path, index_path)


def test_convert_calls_infer_file(vocals_wav, tmp_path):
    output_path = str(tmp_path / "converted.wav")
    with patch("modules.converter.RVCInference") as mock_rvc_cls:
        mock_rvc = MagicMock()
        mock_rvc_cls.return_value = mock_rvc
        vc = VoiceConverter(device="cpu")
        vc._rvc = mock_rvc
        vc.convert(vocals_wav, output_path, f0_up_key=0)
        mock_rvc.infer_file.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_converter.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'modules.converter'`

- [ ] **Step 3: Implement converter.py**

`modules/converter.py`:
```python
from rvc_python.infer import RVCInference


class VoiceConverter:
    """Wraps RVC v2 inference for singing voice conversion."""

    def __init__(self, device: str = "cuda:0"):
        self._rvc = RVCInference(device=device)

    def load_model(self, model_path: str, index_path: str) -> None:
        """Load a trained RVC model and its feature index."""
        self._rvc.load_model(model_path, index_path)

    def convert(
        self,
        input_path: str,
        output_path: str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        filter_radius: int = 3,
    ) -> str:
        """
        Convert vocals audio to target voice.

        Args:
            input_path: Path to source vocals WAV (from Demucs).
            output_path: Path to write converted vocals WAV.
            f0_up_key: Pitch shift in semitones (-12 to +12).
            f0_method: Pitch extraction method. "rmvpe" recommended.
            index_rate: Feature index mixing ratio (0.0–1.0).
            protect: Consonant protection ratio (0.0–0.5).
            rms_mix_rate: RMS volume mix ratio (0.0–1.0).
            filter_radius: Median filter radius for F0 smoothing (int).

        Returns:
            output_path
        """
        self._rvc.infer_file(
            input=input_path,
            output=output_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            filter_radius=filter_radius,
            resample_sr=0,
        )
        return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_converter.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add modules/converter.py tests/test_converter.py
git commit -m "feat: add RVC v2 voice conversion module"
```

---

### Task 5: Training pipeline

**Files:**
- Create: `train.py`
- Create: `tests/test_pipelines.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipelines.py`:
```python
import os
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch, MagicMock


def test_train_returns_model_paths(tmp_path):
    """train() returns (.pth path, .index path) tuple."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.wav").touch()

    with patch("train.run_training") as mock_run:
        mock_run.return_value = (
            str(tmp_path / "model.pth"),
            str(tmp_path / "model.index"),
        )
        from train import train
        model_path, index_path = train(
            model_name="test_model",
            dataset_path=str(dataset_dir),
            models_dir=str(tmp_path),
            total_epoch=1,
            batch_size=1,
        )
    assert model_path.endswith(".pth")
    assert index_path.endswith(".index")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipelines.py::test_train_returns_model_paths -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Implement train.py**

`train.py`:
```python
import os


def run_training(
    model_name: str,
    dataset_path: str,
    models_dir: str,
    total_epoch: int,
    batch_size: int,
    sample_rate: int = 40000,
) -> tuple[str, str]:
    """
    Execute RVC v2 training: preprocess → feature extract → train.

    Args:
        model_name: Identifier for the model (used for file naming).
        dataset_path: Directory containing clean voice WAV files.
        models_dir: Root directory to save trained model files.
        total_epoch: Number of training epochs.
        batch_size: Training batch size.
        sample_rate: Target sample rate for preprocessing (default 40000).

    Returns:
        (model_pth_path, index_path)

    Raises:
        RuntimeError: If rvc_python.train is not available or training fails.
    """
    try:
        from rvc_python.train import train_model
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"rvc_python.train not available: {e}. "
            "Upgrade with: pip install --upgrade infer-rvc-python"
        )

    train_model(
        model_name=model_name,
        dataset_path=dataset_path,
        save_path=models_dir,
        total_epoch=total_epoch,
        batch_size=batch_size,
        sample_rate=sample_rate,
        version="v2",
        pitch_guidance=True,
        gpu_id=0,
    )

    model_pth = os.path.join(models_dir, model_name, f"{model_name}.pth")
    index_path = os.path.join(models_dir, model_name, f"added_{model_name}.index")
    return model_pth, index_path


def train(
    model_name: str,
    dataset_path: str,
    models_dir: str = "models",
    total_epoch: int = 200,
    batch_size: int = 4,
) -> tuple[str, str]:
    """
    Public entry point: train a voice model from audio samples.

    Args:
        model_name: Identifier for this voice model.
        dataset_path: Directory containing clean voice WAV files (10–30 min recommended).
        models_dir: Root directory to save trained model files.
        total_epoch: Number of training epochs (default 200).
        batch_size: Training batch size (default 4, safe for 2080 Ti).

    Returns:
        (model_pth_path, index_path)
    """
    os.makedirs(os.path.join(models_dir, model_name), exist_ok=True)
    return run_training(
        model_name=model_name,
        dataset_path=dataset_path,
        models_dir=models_dir,
        total_epoch=total_epoch,
        batch_size=batch_size,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipelines.py::test_train_returns_model_paths -v
```
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_pipelines.py
git commit -m "feat: add RVC training pipeline"
```

---

### Task 6: Inference pipeline

**Files:**
- Create: `infer.py`
- Modify: `tests/test_pipelines.py` (add inference test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipelines.py`:
```python
def test_infer_returns_output_paths(tmp_path):
    """infer() returns dict with vocals_original, vocals_converted, final_output keys."""
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 300 * t) * 0.4).astype(np.float32)

    song_path = str(tmp_path / "song.wav")
    sf.write(song_path, audio, sr)

    model_path = str(tmp_path / "model.pth")
    index_path = str(tmp_path / "model.index")
    open(model_path, "w").close()
    open(index_path, "w").close()

    output_dir = str(tmp_path / "outputs")
    os.makedirs(output_dir)

    vocals_wav = str(tmp_path / "vocals.wav")
    acc_wav = str(tmp_path / "accompaniment.wav")
    converted_wav = str(tmp_path / "vocals_converted.wav")
    final_wav = str(tmp_path / "final_output.wav")
    for p in [vocals_wav, acc_wav, converted_wav, final_wav]:
        sf.write(p, audio, sr)

    with patch("infer.separate_vocals") as mock_sep, \
         patch("infer.VoiceConverter") as mock_vc_cls, \
         patch("infer.mix_audio") as mock_mix:

        mock_sep.return_value = (vocals_wav, acc_wav)

        mock_vc = MagicMock()
        mock_vc_cls.return_value = mock_vc
        mock_vc.convert.return_value = converted_wav

        mock_mix.return_value = final_wav

        from infer import infer
        result = infer(
            song_path=song_path,
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
        )

    assert "vocals_original" in result
    assert "vocals_converted" in result
    assert "final_output" in result
    assert result["vocals_original"] == vocals_wav
    assert result["vocals_converted"] == converted_wav
    assert result["final_output"] == final_wav
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipelines.py::test_infer_returns_output_paths -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'infer'`

- [ ] **Step 3: Implement infer.py**

`infer.py`:
```python
import os

import torch

from modules.separator import separate_vocals
from modules.converter import VoiceConverter
from modules.mixer import mix_audio


def infer(
    song_path: str,
    model_path: str,
    index_path: str,
    output_dir: str,
    f0_up_key: int = 0,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
) -> dict[str, str]:
    """
    Full inference pipeline: separate vocals → convert voice → mix with accompaniment.

    Args:
        song_path: Path to input song file (MP3/WAV).
        model_path: Path to trained RVC .pth model file.
        index_path: Path to trained RVC .index feature file.
        output_dir: Directory for all intermediate and final output files.
        f0_up_key: Pitch shift in semitones (-12 to +12). Use negative for male→female.
        vocals_volume: Volume multiplier for converted vocals (1.0 = original).
        accompaniment_volume: Volume multiplier for accompaniment (1.0 = original).
        f0_method: Pitch extraction method ("rmvpe" recommended).
        index_rate: Feature index mixing ratio (0.0–1.0).

    Returns:
        Dict with keys:
            "vocals_original": path to separated original vocals
            "vocals_converted": path to RVC-converted vocals
            "final_output": path to final mixed audio
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Separate vocals from accompaniment
    vocals_path, accompaniment_path = separate_vocals(song_path, output_dir)

    # Step 2: Convert vocals to target voice with RVC
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    converter = VoiceConverter(device=device)
    converter.load_model(model_path, index_path)

    converted_path = os.path.join(output_dir, "vocals_converted.wav")
    converter.convert(
        vocals_path,
        converted_path,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        index_rate=index_rate,
    )

    # Step 3: Mix converted vocals with accompaniment
    final_path = os.path.join(output_dir, "final_output.wav")
    mix_audio(
        converted_path,
        accompaniment_path,
        final_path,
        vocals_volume=vocals_volume,
        accompaniment_volume=accompaniment_volume,
    )

    return {
        "vocals_original": vocals_path,
        "vocals_converted": converted_path,
        "final_output": final_path,
    }
```

- [ ] **Step 4: Run all pipeline tests**

```bash
pytest tests/test_pipelines.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add infer.py tests/test_pipelines.py
git commit -m "feat: add full inference pipeline"
```

---

### Task 7: Gradio app

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement app.py**

`app.py`:
```python
import os
import shutil

import gradio as gr

from train import train
from infer import infer

MODELS_DIR = "models"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"

for _d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_available_models() -> list[str]:
    """Return model names that have a .pth file in models/<name>/."""
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for name in sorted(os.listdir(MODELS_DIR)):
        model_dir = os.path.join(MODELS_DIR, name)
        if os.path.isdir(model_dir):
            pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            if pth_files:
                models.append(name)
    return models


def run_training(
    model_name: str,
    audio_files: list,
    total_epoch: int,
    batch_size: int,
    progress=gr.Progress(),
) -> str:
    """Gradio callback for Tab 1 training."""
    if not model_name.strip():
        return "错误：请输入模型名称。"
    if not audio_files:
        return "错误：请上传至少一个声音样本文件。"

    dataset_dir = os.path.join(UPLOADS_DIR, model_name, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    for audio_file in audio_files:
        filename = os.path.basename(audio_file.name)
        shutil.copy(audio_file.name, os.path.join(dataset_dir, filename))

    log_lines = [
        f"模型名称: {model_name}",
        f"数据集目录: {dataset_dir}",
        f"训练轮数: {total_epoch}，批大小: {batch_size}",
        "开始训练...",
    ]

    progress(0.05, desc="开始训练...")
    try:
        model_path, index_path = train(
            model_name=model_name,
            dataset_path=dataset_dir,
            models_dir=MODELS_DIR,
            total_epoch=int(total_epoch),
            batch_size=int(batch_size),
        )
        progress(1.0, desc="训练完成！")
        log_lines += [
            "✅ 训练完成！",
            f"模型路径: {model_path}",
            f"索引路径: {index_path}",
        ]
    except Exception as e:
        log_lines.append(f"❌ 训练失败: {e}")

    return "\n".join(log_lines)


def run_inference(
    model_name: str,
    song_file,
    f0_up_key: int,
    vocals_volume: float,
    accompaniment_volume: float,
    progress=gr.Progress(),
):
    """Gradio callback for Tab 2 inference. Returns (original, converted, final, status)."""
    if not model_name:
        return None, None, None, "错误：请选择模型。"
    if song_file is None:
        return None, None, None, "错误：请上传歌曲文件。"

    model_dir = os.path.join(MODELS_DIR, model_name)
    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    index_files = [f for f in os.listdir(model_dir) if f.endswith(".index")]

    if not pth_files:
        return None, None, None, f"错误：在 {model_dir} 中找不到 .pth 文件"
    if not index_files:
        return None, None, None, f"错误：在 {model_dir} 中找不到 .index 文件"

    model_path = os.path.join(model_dir, pth_files[0])
    index_path = os.path.join(model_dir, index_files[0])
    output_dir = os.path.join(OUTPUTS_DIR, model_name)

    progress(0.1, desc="分离人声中...")
    try:
        results = infer(
            song_path=song_file.name,
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
            f0_up_key=int(f0_up_key),
            vocals_volume=float(vocals_volume),
            accompaniment_volume=float(accompaniment_volume),
        )
        progress(1.0, desc="生成完成！")
        return (
            results["vocals_original"],
            results["vocals_converted"],
            results["final_output"],
            "✅ 生成完成！",
        )
    except Exception as e:
        return None, None, None, f"❌ 生成失败: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="声音克隆翻唱系统") as demo:
        gr.Markdown("# 声音克隆 + 歌声生成系统")

        with gr.Tab("🎤 模型训练"):
            model_name_input = gr.Textbox(label="模型名称", placeholder="例如：jay_chou")
            audio_upload = gr.File(
                label="上传声音样本（支持多选，MP3/WAV，建议10–30分钟干净人声）",
                file_types=[".wav", ".mp3"],
                file_count="multiple",
            )
            with gr.Row():
                epoch_slider = gr.Slider(50, 500, value=200, step=50, label="训练轮数")
                batch_slider = gr.Slider(1, 16, value=4, step=1, label="批大小")
            train_btn = gr.Button("开始训练", variant="primary")
            train_log = gr.Textbox(label="训练日志", lines=12, interactive=False)
            train_btn.click(
                fn=run_training,
                inputs=[model_name_input, audio_upload, epoch_slider, batch_slider],
                outputs=[train_log],
            )

        with gr.Tab("🎵 翻唱生成"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="选择模型",
                    choices=get_available_models(),
                    interactive=True,
                )
                refresh_btn = gr.Button("刷新模型列表")
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_available_models()),
                outputs=[model_dropdown],
            )
            song_upload = gr.File(
                label="上传歌曲（MP3/WAV）",
                file_types=[".wav", ".mp3"],
            )
            f0_slider = gr.Slider(-12, 12, value=0, step=1, label="音调偏移（半音）；男翻女+5~+12，女翻男-5~-12")
            with gr.Row():
                vocal_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="人声音量")
                acc_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="伴奏音量")
            infer_btn = gr.Button("开始生成", variant="primary")
            with gr.Row():
                audio_original = gr.Audio(label="原始人声")
                audio_converted = gr.Audio(label="转换后人声")
                audio_final = gr.Audio(label="最终混音")
            infer_status = gr.Textbox(label="状态", interactive=False)
            infer_btn.click(
                fn=run_inference,
                inputs=[model_dropdown, song_upload, f0_slider, vocal_vol, acc_vol],
                outputs=[audio_original, audio_converted, audio_final, infer_status],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

- [ ] **Step 2: Verify app launches**

```bash
python app.py
```
Expected: Gradio starts at `http://0.0.0.0:7860`. Open in browser. Verify:
- Tab "模型训练" renders with model name input, file upload, sliders, train button, log box.
- Tab "翻唱生成" renders with model dropdown, song upload, pitch slider, volume sliders, generate button, three audio players.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio UI with training and inference tabs"
```

---

### Task 8: Interface smoke tests

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write smoke tests**

`tests/test_smoke.py`:
```python
"""
Smoke tests: verify module imports and interface signatures are consistent.
No model loading or GPU required.
"""
import inspect

from modules.separator import separate_vocals
from modules.mixer import mix_audio
from modules.converter import VoiceConverter
from train import train
from infer import infer


def test_separate_vocals_signature():
    params = list(inspect.signature(separate_vocals).parameters)
    assert "input_path" in params
    assert "output_dir" in params


def test_mix_audio_signature():
    params = list(inspect.signature(mix_audio).parameters)
    assert "vocals_path" in params
    assert "accompaniment_path" in params
    assert "output_path" in params
    assert "vocals_volume" in params
    assert "accompaniment_volume" in params


def test_voice_converter_has_required_methods():
    assert callable(getattr(VoiceConverter, "load_model", None))
    assert callable(getattr(VoiceConverter, "convert", None))


def test_voice_converter_convert_signature():
    params = list(inspect.signature(VoiceConverter.convert).parameters)
    assert "input_path" in params
    assert "output_path" in params
    assert "f0_up_key" in params


def test_train_signature():
    params = list(inspect.signature(train).parameters)
    assert "model_name" in params
    assert "dataset_path" in params
    assert "total_epoch" in params
    assert "batch_size" in params


def test_infer_signature():
    params = list(inspect.signature(infer).parameters)
    assert "song_path" in params
    assert "model_path" in params
    assert "index_path" in params
    assert "output_dir" in params
    assert "f0_up_key" in params
    assert "vocals_volume" in params
    assert "accompaniment_volume" in params


def test_infer_documents_return_keys():
    """infer() docstring must document the three output dict keys."""
    doc = infer.__doc__
    assert "vocals_original" in doc
    assert "vocals_converted" in doc
    assert "final_output" in doc
```

- [ ] **Step 2: Run smoke tests**

```bash
pytest tests/test_smoke.py -v
```
Expected: `7 passed`

- [ ] **Step 3: Run all unit tests (no GPU required)**

```bash
pytest tests/test_smoke.py tests/test_mixer.py tests/test_converter.py tests/test_pipelines.py -v
```
Expected: All pass. (test_separator.py requires GPU + model download — run separately.)

- [ ] **Step 4: Final commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add interface smoke tests for all modules"
```
