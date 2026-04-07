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
