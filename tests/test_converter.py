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
    with patch("modules.converter.BaseLoader"):
        vc = VoiceConverter(device="cpu")
    assert vc is not None


def test_load_model_calls_apply_conf(tmp_path):
    model_path = str(tmp_path / "model.pth")
    index_path = str(tmp_path / "model.index")
    open(model_path, "w").close()
    open(index_path, "w").close()

    with patch("modules.converter.BaseLoader") as mock_cls:
        mock_loader = MagicMock()
        mock_cls.return_value = mock_loader
        vc = VoiceConverter(device="cpu")
        vc.load_model(model_path, index_path)
        mock_loader.apply_conf.assert_called_once()
        call_kwargs = mock_loader.apply_conf.call_args
        assert call_kwargs.kwargs.get("file_model") == model_path or model_path in str(call_kwargs)


def test_convert_calls_loader(vocals_wav, tmp_path):
    output_path = str(tmp_path / "converted.wav")
    with patch("modules.converter.BaseLoader") as mock_cls:
        mock_loader = MagicMock()
        mock_cls.return_value = mock_loader
        mock_loader.output_list = [str(tmp_path / "vocals_edited.wav")]
        # Create a fake output file so shutil.move doesn't fail
        import soundfile as sf
        import numpy as np
        sr = 40000
        sf.write(mock_loader.output_list[0], np.zeros(sr, dtype=np.float32), sr)

        vc = VoiceConverter(device="cpu")
        vc._model_path = "model.pth"
        vc._index_path = "model.index"
        vc.convert(vocals_wav, output_path, f0_up_key=0)
        mock_loader.assert_called_once()
