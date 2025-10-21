from typing import List, Dict

from app.processors.interfaces import ObjectDetectorI
from app.types import DetectedObjectModel

# Optional heavy dependency
try:
    from ultralytics import YOLO  # type: ignore
    _HAVE_YOLO = True
except Exception:
    YOLO = None  # type: ignore
    _HAVE_YOLO = False


class ObjectDetector(ObjectDetectorI):
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.conf = conf
        self._use_yolo = _HAVE_YOLO
        self._model = YOLO(model_name) if self._use_yolo else None

    def detect_in_frames(self, frames: list) -> List[DetectedObjectModel]:
        if not frames:
            return []

        if not self._use_yolo or self._model is None:
            # Lightweight fallback: no detections (keeps pipeline running)
            return []

        names_conf: Dict[str, float] = {}
        results = self._model.predict(frames, conf=self.conf, verbose=False)
        for r in results:
            # r.names: class id to name
            for b in r.boxes:
                cls_id = int(b.cls.item())
                name = r.names[cls_id]
                conf = float(b.conf.item())
                if name not in names_conf or conf > names_conf[name]:
                    names_conf[name] = conf

        return [DetectedObjectModel(name=k, confidence=v) for k, v in names_conf.items()]
