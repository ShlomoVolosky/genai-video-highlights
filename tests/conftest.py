import os
import json
import tempfile
import shutil
import types
import pytest

@pytest.fixture
def tmpfile_mp4():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

@pytest.fixture
def tmpdir_path():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def fake_gemini_text():
    # a valid JSON the selector expects
    return json.dumps([{
        "is_highlight": True,
        "description": "A person enters the room and starts speaking.",
        "summary": "Person speaking.",
        "confidence": 0.87
    }])

@pytest.fixture
def fake_objects():
    return [{"name": "person", "confidence": 0.91}, {"name": "car", "confidence": 0.66}]

@pytest.fixture
def fake_gemini():
    """Mock GeminiClient for tests"""
    class FakeGeminiClient:
        def generate(self, prompt: str) -> str:
            return json.dumps([{
                "is_highlight": True,
                "description": "A person enters the room and starts speaking.",
                "summary": "This is a key moment in the video.",
                "confidence": 0.87
            }])
        
        def embed(self, text: str) -> list[float]:
            # Return a 768-dimensional vector of zeros for testing
            return [0.0] * 768
    
    return FakeGeminiClient()
