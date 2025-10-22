import os
from typing import List
from app.db.repository import Repository
from app.llm.llm_client import UnifiedLLMClient

class ChatService:
    """
    DB-only answering:
    1) If any LLM API key available (GOOGLE_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY): embed question → vector search in pgvector
    2) Else: keyword ILIKE search on description/llm_summary
    3) compose answer from DB rows (no LLM generation)
    """
    def __init__(self, top_k: int = 5):
        self.repo = Repository()
        self.top_k = top_k
        # Try to initialize embedder with any available LLM client
        self.embedder = None
        if os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("CLAUDE_API_KEY"):
            try:
                self.embedder = UnifiedLLMClient()
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM client: {e}")
                self.embedder = None

    def answer(self, question: str) -> tuple[str, List[dict]]:
        # Try vector search if embedder is available
        if self.embedder:
            try:
                q_emb = self.embedder.embed(question)
                rows = self.repo.vector_search(q_emb, top_k=self.top_k)
                # If vector search returns no results, fall back to keyword search
                if not rows:
                    print("⚠️ Vector search returned no results, falling back to keyword search")
                    rows = self.repo.keyword_search(question, top_k=self.top_k)
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}, falling back to keyword search")
                rows = self.repo.keyword_search(question, top_k=self.top_k)
        else:
            # Fallback to keyword search
            rows = self.repo.keyword_search(question, top_k=self.top_k)
        
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
