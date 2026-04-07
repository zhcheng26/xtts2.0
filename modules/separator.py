import os
from pathlib import Path

import torch
import torchaudio
from demucs import pretrained, apply
from demucs.audio import save_audio


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
    model = pretrained.get_model("htdemucs_ft")
    model.to(device)
    model.eval()

    # Load audio and resample to model's expected sample rate
    wav, sr = torchaudio.load(input_path)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    # Ensure stereo (2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.to(device)

    # Apply model: output shape is [sources, channels, samples]
    with torch.no_grad():
        sources = apply.apply_model(model, wav.unsqueeze(0), device=device)[0]

    # sources order matches model.sources list
    stem_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    vocals_idx = stem_names.index("vocals")

    vocals_wav = sources[vocals_idx]
    accompaniment_wav = sum(
        sources[i] for i in range(len(stem_names)) if i != vocals_idx
    )

    vocals_path = os.path.join(output_dir, "vocals.wav")
    accompaniment_path = os.path.join(output_dir, "accompaniment.wav")

    save_audio(vocals_wav.cpu(), vocals_path, samplerate=model.samplerate)
    save_audio(accompaniment_wav.cpu(), accompaniment_path, samplerate=model.samplerate)

    return vocals_path, accompaniment_path
