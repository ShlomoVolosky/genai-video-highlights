import hashlib, subprocess
from pathlib import Path
from typing import Tuple, Optional

from app.processors.interfaces import VideoFetcher


class VideoDownloader(VideoFetcher):
    def __init__(self, out_dir: str = "data/videos"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _hash_path(self, path: str) -> str:
        return hashlib.sha1(path.encode()).hexdigest()[:16]

    def fetch(self, source: str) -> Tuple[str, Optional[str]]:
        """
        Process local video files only.
        Args:
            source: Path to local video file
        Returns:
            Tuple of (video_path, video_uid)
        """
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        
        # Generate unique ID based on file path
        uid = self._hash_path(str(src.resolve()))
        
        # Copy to processing directory if not already there
        dst = self.out_dir / f"{uid}{src.suffix if src.suffix else '.mp4'}"
        if str(src.resolve()) != str(dst.resolve()):
            subprocess.run(["cp", str(src), str(dst)], check=True)
        
        return str(dst), uid
