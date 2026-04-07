"""
Smoke tests: verify module imports and interface signatures are consistent.
No model loading or GPU required.
"""
import inspect
import sys
from unittest.mock import MagicMock

# Mock demucs.api if not available (before importing separator).
# Do NOT clean it up—other tests may depend on this mock persisting.
try:
    import demucs.api
except ImportError:
    sys.modules["demucs.api"] = MagicMock()

from modules.separator import separate_vocals
from modules.mixer import mix_audio
from modules.converter import VoiceConverter
from train import train


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
    from infer import infer
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
    from infer import infer
    doc = infer.__doc__
    assert "vocals_original" in doc
    assert "vocals_converted" in doc
    assert "final_output" in doc
