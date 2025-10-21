import types
from app.processors.audio_transcriber import AudioTranscriber

def test_transcribe_monkeypatch(monkeypatch):
    # Mock whisper.load_model and model.transcribe
    class FakeModel:
        def transcribe(self, wav, fp16=False):
            return {"text": "hello world", "segments": [{"end": 3.2}]}
    
    import app.processors.audio_transcriber as mod
    
    # Create a fake whisper module if it doesn't exist
    if mod.whisper is None:
        fake_whisper = types.ModuleType('whisper')
        fake_whisper.load_model = lambda name: FakeModel()
        monkeypatch.setattr(mod, 'whisper', fake_whisper)
        monkeypatch.setattr(mod, '_HAVE_WHISPER', True)
    else:
        monkeypatch.setattr(mod.whisper, "load_model", lambda name: FakeModel())
    
    # Mock subprocess to avoid actual ffmpeg calls
    import subprocess
    class FakeCompletedProcess:
        def __init__(self):
            self.stdout = b'{"format": {"duration": "3.2"}}'
    
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: b'{"format": {"duration": "3.2"}}')
    monkeypatch.setattr("tempfile.mktemp", lambda suffix="": "/fake/path.wav")
    monkeypatch.setattr("os.path.exists", lambda path: True)  # Make the file appear to exist
    
    at = AudioTranscriber(model_name="base")
    text, dur = at.transcribe("anything.mp4")
    assert text == "hello world"
    assert dur == 3.2
