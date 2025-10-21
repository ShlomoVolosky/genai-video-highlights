from typing import List
from app.db.repository import Repository
from app.llm.gemini_client import GeminiClient

class ChatService:
    """
    DB-only answering:
    1) embed question (Gemini embeddings only)
    2) vector search in pgvector (top-k highlights)
    3) compose answer from DB rows (no LLM generation)
    """
    def __init__(self, top_k: int = 5):
        self.repo = Repository()
        self.embedder = GeminiClient()
        self.top_k = top_k

    def answer(self, question: str) -> tuple[str, List[dict]]:
        q_emb = self.embedder.embed(question)
        rows = self.repo.vector_search(q_emb, top_k=self.top_k)  # list of dicts with score
        if not rows:
            return "I couldn't find relevant highlights for that question.", []

        # Build a coherent, **DB-only** answer.
        # Order by video then timestamp to read like a narrative.
        rows_sorted = sorted(rows, key=lambda r: (r["video_id"], r["ts_start_sec"]))
        sentences = []
        for r in rows_sorted:
            t = f"[{r['ts_start_sec']}s–{r['ts_end_sec']}s]"
            s = r.get("llm_summary") or r.get("description", "")
            if s:
                sentences.append(f"{t} {s}")

        final = " ".join(sentences) if sentences else "No highlight text available."
        return final, rows_sorted
