from app.main import VideoProcessor
from app.types import DetectedObjectModel, HighlightModel


def test_video_processor_flow(mocker, fake_gemini, tmpdir):
    vp = VideoProcessor()

    # Mock downloader -> returns a "video path" and uid
    mocker.patch.object(vp.downloader, "fetch", return_value=(f"{tmpdir}/v.mp4", "vid123"))

    # Mock transcriber -> transcript, duration
    mocker.patch.object(vp.transcriber, "transcribe", return_value=("hello", 30.0))

    # Mock scenes -> one scene [0,5]
    mocker.patch.object(vp.scenes, "detect_scenes", return_value=[(0, 5)])

    # Mock frames -> not important, empty list ok
    mocker.patch.object(vp.sampler, "sample", return_value=[])

    # Mock objects -> one person
    mocker.patch.object(vp.objects, "detect_in_frames",
                        return_value=[DetectedObjectModel(name="person", confidence=0.9)])

    # Inject fake gemini client into selector
    vp.selector.client = fake_gemini

    # Spy repo methods
    mock_upsert = mocker.patch.object(vp.repo, "upsert_video", return_value=type("V", (), {"id": 1})())
    mock_add = mocker.patch.object(vp.repo, "add_highlights", return_value=[1])

    video, highlights = vp.process("https://www.youtube.com/watch?v=abc")
    assert len(highlights) >= 1
    assert isinstance(highlights[0], HighlightModel)
    mock_upsert.assert_called_once()
    mock_add.assert_called_once()
