import os
import shutil

try:
    from infer_rvc_python.main import BaseLoader
except ImportError:
    BaseLoader = None


class VoiceConverter:
    """Wraps infer_rvc_python BaseLoader for singing voice conversion."""

    def __init__(self, device: str = "cuda:0"):
        only_cpu = device == "cpu"
        self._loader = BaseLoader(only_cpu=only_cpu)
        self._tag = "model"

    def load_model(self, model_path: str, index_path: str) -> None:
        """Load a trained RVC model and its feature index."""
        self._model_path = model_path
        self._index_path = index_path
        self._loader.apply_conf(
            tag=self._tag,
            file_model=model_path,
            file_index=index_path,
        )

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
        self._loader.apply_conf(
            tag=self._tag,
            file_model=self._model_path,
            file_index=self._index_path,
            pitch_algo=f0_method,
            pitch_lvl=f0_up_key,
            index_influence=index_rate,
            consonant_breath_protection=protect,
            envelope_ratio=rms_mix_rate,
            respiration_median_filtering=filter_radius,
        )

        self._loader(
            audio_files=[input_path],
            tag_list=[self._tag],
            type_output="wav",
        )

        # infer_rvc_python writes to {dirname}/{stem}_edited.wav
        auto_output = self._loader.output_list[0]
        if auto_output != output_path:
            shutil.move(auto_output, output_path)

        return output_path
