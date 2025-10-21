import tempfile, subprocess, json
from typing import Tuple, Optional
import os

from app.processors.interfaces import Transcriber


def _probe_duration_ffprobe(path_or_url: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path_or_url,
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    info = json.loads(out)
    dur = 0.0
    if "format" in info and "duration" in info["format"]:
        try: dur = float(info["format"]["duration"])
        except Exception: dur = 0.0
    if not dur and "streams" in info:
        for s in info["streams"]:
            if "duration" in s:
                try: dur = max(dur, float(s["duration"]))
                except Exception: pass
    return float(dur or 0.0)


class AudioTranscriber(Transcriber):
    """
    Simple transcriber for local video files.
    Returns empty transcript but extracts duration using ffprobe.
    """
    def __init__(self, model_name: str = "base"):
        # No heavy dependencies - just use ffprobe for duration
        pass

    def transcribe(self, video_path: str) -> Tuple[str, float]:
        """
        Extract duration from local video file.
        Returns empty transcript and video duration.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get duration using ffprobe
        duration = _probe_duration_ffprobe(video_path)
        
        # Return empty transcript with duration
        # In a real implementation, you could integrate with speech-to-text services
        return "", duration
