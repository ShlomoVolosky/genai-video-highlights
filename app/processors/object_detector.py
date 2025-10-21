from typing import List
from ultralytics import YOLO
from app.processors.interfaces import ObjectDetectorI
from app.types import DetectedObjectModel


class ObjectDetector(ObjectDetectorI):
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect_in_frames(self, frames: list) -> List[DetectedObjectModel]:
        if not frames:
            return []
        names_conf: dict[str, float] = {}
        results = self.model.predict(frames, conf=self.conf, verbose=False)
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                name = r.names[cls_id]
                conf = float(b.conf.item())
                if name not in names_conf or conf > names_conf[name]:
                    names_conf[name] = conf
        return [DetectedObjectModel(name=k, confidence=v) for k, v in names_conf.items()]
