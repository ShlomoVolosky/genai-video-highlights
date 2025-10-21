import cv2
from app.processors.interfaces import FrameProvider


class FrameSampler(FrameProvider):
    def __init__(self, every_sec: float = 1.5):
        if every_sec <= 0:
            raise ValueError("every_sec must be > 0")
        self.every_sec = every_sec

    def sample(self, video_path: str, start_sec: int, end_sec: int) -> list:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = []
        t = float(start_sec)
        while t <= float(end_sec):
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            t += self.every_sec
        cap.release()
        return frames
