from abc import ABC, abstractmethod
from typing import Tuple, List
from app.types import DetectedObjectModel


class VideoFetcher(ABC):
    @abstractmethod
    def fetch(self, source: str) -> Tuple[str, str | None]:
        """Return (video_path, video_uid)."""


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, video_path: str) -> Tuple[str, float]:
        """Return (transcript_text, duration_sec)."""


class SceneFinder(ABC):
    @abstractmethod
    def detect_scenes(self, video_path: str) -> list[tuple[int, int]]:
        """Return list of (start_sec, end_sec)."""


class FrameProvider(ABC):
    @abstractmethod
    def sample(self, video_path: str, start_sec: int, end_sec: int) -> List:
        """Return list of frames (numpy arrays)."""


class ObjectDetectorI(ABC):
    @abstractmethod
    def detect_in_frames(self, frames: List) -> List[DetectedObjectModel]:
        """Return objects detected across frames."""
