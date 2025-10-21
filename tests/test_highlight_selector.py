from app.llm.highlight_selector import HighlightSelector
from app.types import DetectedObjectModel


def test_highlight_selector_basic(fake_gemini):
    selector = HighlightSelector(fake_gemini)
    seg = (0, 5)
    transcript = "hello world"
    objects = [DetectedObjectModel(name="person", confidence=0.8)]
    hl = selector.analyze_segment(seg, transcript, objects)
    assert hl is not None
    assert hl.ts_start_sec == 0
    assert hl.ts_end_sec == 5
    assert "key moment" in hl.llm_summary.lower()
