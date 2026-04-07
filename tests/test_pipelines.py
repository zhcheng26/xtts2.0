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
