import tempfile, subprocess, json, urllib.request
from typing import Tuple, Optional
import os

from app.processors.interfaces import Transcriber

# Optional Whisper (CPU PyTorch). If present, we use it for local files.
try:
    import whisper  # type: ignore
    _HAVE_WHISPER = True
except Exception:
    whisper = None  # type: ignore
    _HAVE_WHISPER = False

# Optional yt-dlp for captions on YouTube URLs
try:
    import yt_dlp  # type: ignore
    _HAVE_YT = True
except Exception:
    yt_dlp = None  # type: ignore
    _HAVE_YT = False


def _is_youtube(src: str) -> bool:
    s = src.lower()
    return ("youtube.com" in s) or ("youtu.be" in s)


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


def _vtt_to_text(vtt: str) -> str:
    """
    Minimal WebVTT to plain text: drop headers, timestamps, cues.
    """
    lines = []
    for line in vtt.splitlines():
        l = line.strip()
        if not l:
            continue
        if l.startswith("WEBVTT") or "-->" in l or l.isdigit():
            continue
        # remove possible cue settings like <c>…</c>
        l = l.replace("<c>", "").replace("</c>", "")
        lines.append(l)
    return "\n".join(lines).strip()


class AudioTranscriber(Transcriber):
    """
    Transcribes using:
    - Whisper (if installed) for local files
    - YouTube captions for URLs (no Whisper needed)
    Falls back to empty text + probed duration if neither path is available.
    """
    def __init__(self, model_name: str = "base"):
        self._use_whisper = _HAVE_WHISPER
        self._model = whisper.load_model(model_name) if self._use_whisper else None
        self._have_yt = _HAVE_YT

    def _extract_audio(self, video_path: str) -> str:
        wav = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return wav

    def _youtube_captions(self, url: str) -> tuple[str, float]:
        if not self._have_yt:
            return "", 0.0
        # Use yt-dlp to inspect available captions (subtitles or auto)
        ydl_opts = {"skip_download": True, "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            info = ydl.extract_info(url, download=False)
            duration = float(info.get("duration", 0.0) or 0.0)
            captions = info.get("subtitles") or info.get("automatic_captions") or {}
            # prefer English if available; else first available
            track = None
            for pref in ("en", "en-US", "en-GB"):
                if pref in captions:
                    track = captions[pref]
                    break
            if not track and captions:
                # pick any language
                track = next(iter(captions.values()))
            if not track:
                return "", duration

            # choose a text-friendly format url (webvtt if present)
            # track is a list of dicts like [{'ext':'vtt','url':...}, ...]
            vtt_url = None
            for item in track:
                if item.get("ext") in ("vtt", "srv1", "srv2", "srv3", "ttml", "srt"):
                    vtt_url = item.get("url")
                    if item.get("ext") == "vtt":
                        break
            if not vtt_url:
                return "", duration

            try:
                vtt_bytes = urllib.request.urlopen(vtt_url, timeout=15).read()
                text = _vtt_to_text(vtt_bytes.decode("utf-8", errors="ignore"))
                return text, duration
            except Exception:
                return "", duration

    def transcribe(self, video_path_or_url: str) -> Tuple[str, float]:
        # If it looks like a YouTube URL -> try captions first (lightweight, no Whisper)
        if _is_youtube(video_path_or_url):
            text, dur = self._youtube_captions(video_path_or_url)
            if text:
                return text, dur or 0.0
            # If captions unavailable, fall through (we won't download; just probe duration)
            return "", dur or 0.0

        # Otherwise we assume it's a local file path
        if os.path.exists(video_path_or_url) and self._use_whisper and self._model is not None:
            wav = self._extract_audio(video_path_or_url)
            result = self._model.transcribe(wav, fp16=False)
            text = (result.get("text") or "").strip()
            duration = float(result["segments"][-1]["end"]) if result.get("segments") else _probe_duration_ffprobe(video_path_or_url)
            return text, duration

        # No Whisper (or path missing): just return probed duration
        duration = _probe_duration_ffprobe(video_path_or_url)
        return "", duration
