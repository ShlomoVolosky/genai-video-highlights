import numpy as np

def test_scene_detector(monkeypatch):
    from app.processors.scene_detector import SceneDetector
    # Mock scenedetect.detect to return a list of "scenes" objects with get_seconds()
    class FakeFrame:
        def __init__(self, s): self._s = s
        def get_seconds(self): return self._s
    def fake_detect(video=None, detector=None):
        return [(FakeFrame(0.2), FakeFrame(3.8)), (FakeFrame(5.0), FakeFrame(9.4))]
    import app.processors.scene_detector as sd_mod
    monkeypatch.setattr(sd_mod, "detect", fake_detect)
    sd = SceneDetector()
    segs = sd.detect_scenes("video.mp4")
    assert segs == [(0, 4), (5, 10)]

def test_frame_sampler(monkeypatch, tmpfile_mp4):
    from app.processors.frame_sampler import FrameSampler
    # Mock cv2.VideoCapture to deliver 3 frames
    class FakeCap:
        def __init__(self, p): self._pos = 0
        def get(self, prop): return 25.0  # fps
        def set(self, prop, value): self._pos = value
        def read(self):
            # produce a synthetic image
            if self._pos > 1000: return False, None
            return True, np.zeros((10, 10, 3), dtype=np.uint8)
        def release(self): pass
    import app.processors.frame_sampler as fs_mod
    monkeypatch.setattr(fs_mod, "cv2", type("M",(),{"VideoCapture": lambda p: FakeCap(p), "CAP_PROP_FPS": 5, "CAP_PROP_POS_FRAMES": 1}))
    sampler = FrameSampler(every_sec=1.5)
    frames = sampler.sample(tmpfile_mp4, 0, 3)
    assert len(frames) >= 2

def test_object_detector(monkeypatch):
    from app.processors.object_detector import ObjectDetector
    # Mock YOLO().predict to emulate detections
    class FakeBox:
        def __init__(self, cls_id, conf): self.cls = FakeT(cls_id); self.conf = FakeT(conf)
    class FakeT:
        def __init__(self,v): self._v=v
        def item(self): return self._v
    class FakeRes:
        names = {0: "person", 1: "car"}
        def __init__(self): self.boxes=[FakeBox(0, 0.9), FakeBox(1, 0.7)]
    class FakeYOLO:
        def __init__(self, model): pass
        def predict(self, frames, conf=0.35, verbose=False): return [FakeRes()]
    import app.processors.object_detector as od_mod
    monkeypatch.setattr(od_mod, "YOLO", FakeYOLO)
    det = ObjectDetector("yolov8n.pt", conf=0.35)
    objs = det.detect_in_frames([1,2,3])
    names = sorted([o.name for o in objs])
    assert names == ["car","person"]
