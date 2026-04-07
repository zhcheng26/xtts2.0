import math
from pydub import AudioSegment


def mix_audio(
    vocals_path: str,
    accompaniment_path: str,
    output_path: str,
    vocals_volume: float = 1.0,
    accompaniment_volume: float = 1.0,
) -> str:
    """
    Mix vocals and accompaniment into a single output file.

    Args:
        vocals_path: Path to vocals WAV file.
        accompaniment_path: Path to accompaniment WAV file.
        output_path: Path to write the mixed output WAV file.
        vocals_volume: Volume multiplier for vocals (1.0 = unchanged, 0.0 = silent).
        accompaniment_volume: Volume multiplier for accompaniment (1.0 = unchanged, 0.0 = silent).

    Returns:
        output_path
    """
    vocals = AudioSegment.from_file(vocals_path)
    accompaniment = AudioSegment.from_file(accompaniment_path)

    if vocals_volume == 0.0:
        vocals = AudioSegment.silent(duration=len(vocals), frame_rate=vocals.frame_rate)
    elif vocals_volume != 1.0:
        db_change = 20 * math.log10(vocals_volume)
        vocals = vocals + db_change

    if accompaniment_volume == 0.0:
        accompaniment = AudioSegment.silent(duration=len(accompaniment), frame_rate=accompaniment.frame_rate)
    elif accompaniment_volume != 1.0:
        db_change = 20 * math.log10(accompaniment_volume)
        accompaniment = accompaniment + db_change

    mixed = accompaniment.overlay(vocals)
    mixed.export(output_path, format="wav")
    return output_path
