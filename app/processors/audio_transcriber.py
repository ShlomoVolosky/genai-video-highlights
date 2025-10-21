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
    Audio transcriber that extracts speech-to-text from video files.
    Uses OpenAI Whisper for speech recognition to detect "people speaking".
    """
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        # Try to load faster-whisper, fallback gracefully if not available
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.has_whisper = True
            print(f"✅ Faster-Whisper model '{model_name}' loaded for speech-to-text")
        except Exception as e:
            print(f"⚠️ Faster-Whisper not available: {e}")
            print("   Installing: pip install faster-whisper")
            self.whisper_model = None
            self.has_whisper = False

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video for transcription"""
        import tempfile
        wav_path = tempfile.mktemp(suffix=".wav")
        try:
            # Extract audio using ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, 
                "-vn", "-ac", "1", "-ar", "16000", 
                wav_path
            ], check=True, capture_output=True)
            return wav_path
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Audio extraction failed: {e}")
            return None

    def transcribe(self, video_path: str) -> Tuple[str, float]:
        """
        Extract speech-to-text transcript and duration from video.
        This enables detection of 'people speaking' as required.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get duration using ffprobe
        duration = _probe_duration_ffprobe(video_path)
        
        # Try to get transcript using Whisper
        if self.has_whisper and self.whisper_model:
            try:
                # Extract audio for transcription
                audio_path = self._extract_audio(video_path)
                if audio_path and os.path.exists(audio_path):
                    # Transcribe with faster-whisper
                    segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments]).strip()
                    
                    # Clean up temporary audio file
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                    
                    if transcript:
                        print(f"🎤 Speech detected: {len(transcript)} characters")
                        return transcript, duration
                    else:
                        print("🔇 No speech detected in audio")
                        return "", duration
                        
            except Exception as e:
                print(f"⚠️ Transcription failed: {e}")
        
        # Fallback: return empty transcript but correct duration
        print("📝 No speech-to-text available, using duration only")
        return "", duration
