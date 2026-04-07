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
