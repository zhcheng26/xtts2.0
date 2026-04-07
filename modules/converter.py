try:
    from rvc_python.infer import RVCInference
except ImportError:
    RVCInference = None


class VoiceConverter:
    """Wraps RVC v2 inference for singing voice conversion."""

    def __init__(self, device: str = "cuda:0"):
        self._rvc = RVCInference(device=device)

    def load_model(self, model_path: str, index_path: str) -> None:
        """Load a trained RVC model and its feature index."""
        self._rvc.load_model(model_path, index_path)

    def convert(
        self,
        input_path: str,
        output_path: str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        filter_radius: int = 3,
    ) -> str:
        """
        Convert vocals audio to target voice.

        Args:
            input_path: Path to source vocals WAV (from Demucs).
            output_path: Path to write converted vocals WAV.
            f0_up_key: Pitch shift in semitones (-12 to +12).
            f0_method: Pitch extraction method. "rmvpe" recommended.
            index_rate: Feature index mixing ratio (0.0–1.0).
            protect: Consonant protection ratio (0.0–0.5).
            rms_mix_rate: RMS volume mix ratio (0.0–1.0).
            filter_radius: Median filter radius for F0 smoothing (int).

        Returns:
            output_path
        """
        self._rvc.infer_file(
            input=input_path,
            output=output_path,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            filter_radius=filter_radius,
            resample_sr=0,
        )
        return output_path
