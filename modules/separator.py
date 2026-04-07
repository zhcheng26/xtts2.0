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
