import hashlib, subprocess
from pathlib import Path
from typing import Tuple, Optional
import yt_dlp

from app.processors.interfaces import VideoFetcher


class VideoDownloader(VideoFetcher):
    def __init__(self, out_dir: str = "data/videos"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _hash_path(self, path: str) -> str:
        return hashlib.sha1(path.encode()).hexdigest()[:16]

    def _is_youtube(self, src: str) -> bool:
        return "youtube.com" in src or "youtu.be" in src

    def fetch(self, source: str) -> Tuple[str, Optional[str]]:
        if self._is_youtube(source):
            ydl_opts = {
                "format": "mp4[height<=720]/mp4/best",
                "outtmpl": str(self.out_dir / "%(id)s.%(ext)s"),
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=True)
                ytid = info.get("id")
                filename = self.out_dir / f"{ytid}.mp4"
                return str(filename), ytid
        else:
            src = Path(source)
            if not src.exists():
                raise FileNotFoundError(source)
            uid = self._hash_path(str(src.resolve()))
            dst = self.out_dir / f"{uid}{src.suffix if src.suffix else '.mp4'}"
            if str(src.resolve()) != str(dst.resolve()):
                subprocess.run(["cp", str(src), str(dst)], check=True)
            return str(dst), uid
