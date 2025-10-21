import math
from scenedetect import detect, ContentDetector
from app.processors.interfaces import SceneFinder


class SceneDetector(SceneFinder):
    def __init__(self, threshold: int = 27):
        self.threshold = threshold

    def detect_scenes(self, video_path: str) -> list[tuple[int, int]]:
        scenes = detect(video_path, ContentDetector(threshold=self.threshold))
        out: list[tuple[int, int]] = []
        for s in scenes:
            start_sec = math.floor(s[0].get_seconds())
            end_sec = math.ceil(s[1].get_seconds())
            if end_sec > start_sec:
                out.append((start_sec, end_sec))
        return out
