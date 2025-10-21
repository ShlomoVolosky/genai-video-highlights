import os
import types
import pytest
from app.processors.video_downloader import VideoDownloader

def test_fetch_local_file(tmpfile_mp4, tmpdir_path):
    vd = VideoDownloader(out_dir=tmpdir_path)
    out, uid = vd.fetch(tmpfile_mp4)
    assert os.path.exists(out)
    assert uid is not None
    assert out.endswith(".mp4")

def test_fetch_youtube_mocks(monkeypatch, tmpdir_path):
    # Mock yt_dlp so we don't download anything
    class FakeYDL:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=True):
            return {"id": "FAKEID123"}
    import app.processors.video_downloader as vd_mod
    monkeypatch.setattr(vd_mod, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    vd = VideoDownloader(out_dir=tmpdir_path)
    out, uid = vd.fetch("https://www.youtube.com/watch?v=abc")
    assert uid == "FAKEID123"
    assert out.endswith("FAKEID123.mp4")
