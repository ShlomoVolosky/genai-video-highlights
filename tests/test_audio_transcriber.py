import types
from app.processors.audio_transcriber import AudioTranscriber

def test_transcribe_monkeypatch(monkeypatch):
    # Mock whisper.load_model and model.transcribe
    class FakeModel:
        def transcribe(self, wav, fp16=False):
            return {"text": "hello world", "segments": [{"end": 3.2}]}
    import app.processors.audio_transcriber as mod
    monkeypatch.setattr(mod.whisper, "load_model", lambda name: FakeModel())
    at = AudioTranscriber(model_name="base")
    text, dur = at.transcribe("anything.mp4")
    assert text == "hello world"
    assert dur == 3.2
