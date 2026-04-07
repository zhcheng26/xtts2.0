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
