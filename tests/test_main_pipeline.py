from types import SimpleNamespace
from app.main import VideoProcessor

def test_main_pipeline_mocks(monkeypatch):
    vp = VideoProcessor()

    # Mock downloader
    monkeypatch.setattr(vp.downloader, "fetch", lambda src: ("video.mp4", "YID"))

    # Mock transcriber
    monkeypatch.setattr(vp.transcriber, "transcribe", lambda p: ("hello there", 8.0))

    # Mock scenes (two segments)
    monkeypatch.setattr(vp.scenes, "detect_scenes", lambda p: [(0,3),(4,7)])

    # Mock frames + objects
    monkeypatch.setattr(vp.sampler, "sample", lambda p, s, e: [1,2])
    monkeypatch.setattr(vp.objects, "detect_in_frames", lambda frames: [SimpleNamespace(name="person", confidence=0.9)])

    # Mock selector (returns a HighlightModel-like object)
    class FakeHL:
        def __init__(self, s,e):
            self.ts_start_sec=s; self.ts_end_sec=e
            self.description="desc"; self.llm_summary="sum"
            self.confidence=0.8; self.objects=[]
            self.embedding=None
    monkeypatch.setattr(vp.selector, "analyze_segment", lambda seg, t, o: FakeHL(*seg))
    monkeypatch.setattr(vp.selector, "embed_desc", lambda text: [0.1]*768)

    captured = {"added": None}
    def fake_add(video_id, highs):
        captured["added"] = (video_id, highs)
        return [1,2]
    monkeypatch.setattr(vp.repo, "add_highlights", fake_add)

    video, highs = vp.process("https://youtube.com/watch?v=ABC")
    assert video.id > 0  # real repo.upsert_video returns VideoRecord
    assert len(highs) == 2
    assert captured["added"][0] == video.id
