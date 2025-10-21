import json
from typing import Tuple, Optional
from app.llm.gemini_client import GeminiClient
from app.types import HighlightModel, DetectedObjectModel

SYSTEM_PROMPT = """You are an expert video analyst.
Given:
- a scene time range in seconds
- speech transcript snippets (if any) from that range
- objects detected (e.g., person, car, dog, fire, explosion)

Pick if this segment is an IMPORTANT HIGHLIGHT (true/false).
If true, write:
- a vivid but factual description (1-3 sentences)
- a 1-sentence summary
- a confidence score 0..1.

Return strict JSON list of objects:
{
  "is_highlight": true|false,
  "description": "...",
  "summary": "...",
  "confidence": 0.0
}
"""

class HighlightSelector:
    def __init__(self, client: GeminiClient):
        self.client = client

    def analyze_segment(
        self,
        seg: Tuple[int, int],
        transcript: str,
        objects: list[DetectedObjectModel],
    ) -> Optional[HighlightModel]:
        start, end = seg
        obj_txt = ", ".join([f"{o.name}({o.confidence:.2f})" for o in objects]) if objects else "none"
        snippet = transcript[:1200] if transcript else ""
        user_prompt = f"""
Scene: {start}s to {end}s
Objects: {obj_txt}
Transcript excerpt (may be empty): {snippet}

{SYSTEM_PROMPT}
Return JSON only.
"""
        raw = self.client.generate(user_prompt)
        try:
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
            for it in items:
                if it.get("is_highlight"):
                    return HighlightModel(
                        ts_start_sec=start,
                        ts_end_sec=end,
                        description=(it.get("description") or "").strip(),
                        llm_summary=(it.get("summary") or "").strip(),
                        confidence=float(it.get("confidence", 0.6)),
                        objects=objects,
                    )
        except Exception:
            # Heuristic fallback (only if some objects exist)
            if objects:
                return HighlightModel(
                    ts_start_sec=start,
                    ts_end_sec=end,
                    description=f"Notable activity with objects: {', '.join(o.name for o in objects)}.",
                    llm_summary="Notable visual activity.",
                    confidence=0.55,
                    objects=objects,
                )
        return None

    def embed_desc(self, text: str) -> list[float]:
        return self.client.embed(text)
