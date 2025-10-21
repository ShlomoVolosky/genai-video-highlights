from app.llm.highlight_selector import HighlightSelector
from app.types import DetectedObjectModel

def test_selector_json_path(monkeypatch, fake_gemini_text):
    class FakeClient:
        def generate(self, prompt): return fake_gemini_text
        def embed(self, text): return [0.1, 0.2, 0.3]  # not used here
    sel = HighlightSelector(FakeClient())
    hl = sel.analyze_segment((0,5), "hello", [DetectedObjectModel(name="person", confidence=0.9)])
    assert hl is not None
    assert hl.description
    assert 0 <= (hl.confidence or 0) <= 1

def test_selector_fallback_without_valid_json(monkeypatch):
    class FakeClient:
        def generate(self, prompt): return "nonsense"
        def embed(self, text): return [0.1, 0.2]
    sel = HighlightSelector(FakeClient())
    hl = sel.analyze_segment((0,5), "", [DetectedObjectModel(name="car", confidence=0.6)])
    assert hl is not None
    assert "car" in hl.description.lower()
