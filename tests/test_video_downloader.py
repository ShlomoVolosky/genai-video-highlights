import os
import pytest
from app.processors.video_downloader import VideoDownloader

def test_fetch_local_file(tmpfile_mp4, tmpdir_path):
    vd = VideoDownloader(out_dir=tmpdir_path)
    out, uid = vd.fetch(tmpfile_mp4)
    assert os.path.exists(out)
    assert uid is not None
    assert out.endswith(".mp4")

def test_fetch_nonexistent_file():
    vd = VideoDownloader()
    with pytest.raises(FileNotFoundError):
        vd.fetch("nonexistent_file.mp4")
