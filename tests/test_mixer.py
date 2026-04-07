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
