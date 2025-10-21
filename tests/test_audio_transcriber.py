from app.processors.audio_transcriber import AudioTranscriber

def test_transcribe_simple(monkeypatch):
    # Mock subprocess to avoid actual ffmpeg calls
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: b'{"format": {"duration": "3.2"}}')
    monkeypatch.setattr("os.path.exists", lambda path: True)  # Make the file appear to exist
    
    at = AudioTranscriber()
    text, dur = at.transcribe("test_video.mp4")
    
    # The simplified transcriber returns empty text and probed duration
    assert text == ""
    assert dur == 3.2
