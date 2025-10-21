import tempfile, subprocess
from typing import Tuple
import whisper

from app.processors.interfaces import Transcriber


class AudioTranscriber(Transcriber):
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def _extract_audio(self, video_path: str) -> str:
        wav = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return wav

    def transcribe(self, video_path: str) -> Tuple[str, float]:
        wav = self._extract_audio(video_path)
        result = self.model.transcribe(wav, fp16=False)
        text = (result.get("text") or "").strip()
        duration = float(result.get("segments", [])[-1]["end"]) if result.get("segments") else 0.0
        return text, duration
