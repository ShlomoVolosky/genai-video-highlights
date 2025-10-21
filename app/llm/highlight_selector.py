import json
from typing import Tuple, Optional
from app.llm.llm_client import UnifiedLLMClient
from app.types import HighlightModel, DetectedObjectModel

SYSTEM_PROMPT = """You are an expert video analyst specializing in identifying important moments.

IMPORTANT HIGHLIGHT CRITERIA - A scene is a highlight if it contains ANY of these:

🎤 PEOPLE SPEAKING:
- Any speech/dialogue detected in transcript
- Conversations, presentations, interviews
- Narration or commentary

🚗 VEHICLE MOVEMENT:
- Cars, trucks, motorcycles, bicycles in motion
- Vehicle-related activity or scenes
- Transportation events

💥 EXPLOSIVE/DRAMATIC EVENTS:
- Explosions, fires, crashes
- Sudden movements or action
- High-energy visual events

👥 HUMAN ACTIVITY:
- People performing actions
- Social interactions
- Notable human behavior

🎬 VISUAL INTEREST:
- Scene changes or transitions
- Multiple objects present
- Dynamic visual content

Given:
- Scene time range in seconds
- Speech transcript (if people are speaking)
- Objects detected (people, vehicles, etc.)

DECISION RULE: If ANY of the above criteria are met, mark as highlight=true.
Be GENEROUS in identifying highlights - err on the side of inclusion.

Return JSON:
{
  "is_highlight": true|false,
  "description": "Detailed description of what makes this moment interesting",
  "summary": "One sentence summary",
  "confidence": 0.0
}
"""

class HighlightSelector:
    def __init__(self, client: UnifiedLLMClient):
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
