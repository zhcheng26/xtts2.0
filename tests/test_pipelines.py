import os
import sys
import numpy as np
import soundfile as sf
import pytest
from unittest.mock import patch, MagicMock, Mock


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
    converted_wav = str(output_dir + "/vocals_converted.wav")
    final_wav = str(output_dir + "/final_output.wav")
    for p in [vocals_wav, acc_wav]:
        sf.write(p, audio, sr)

    # Mock demucs.api before importing infer
    mock_api = Mock()
    sys.modules["demucs.api"] = mock_api

    try:
        with patch("modules.separator.separate_vocals") as mock_sep, \
             patch("modules.converter.VoiceConverter") as mock_vc_cls, \
             patch("modules.mixer.mix_audio") as mock_mix:

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
    finally:
        # Clean up the mock
        if "demucs.api" in sys.modules:
            del sys.modules["demucs.api"]
