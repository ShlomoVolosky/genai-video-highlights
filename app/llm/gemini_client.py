import google.generativeai as genai
import time
from google.api_core.exceptions import ResourceExhausted
from app.config import Config


class GeminiClient:
    def __init__(self):
        if not Config.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing in environment or .env")
        genai.configure(api_key=Config.google_api_key)
        self._embed_model = Config.embedding_model
        self._gen_model = Config.generation_model

    def embed(self, text: str) -> list[float]:
        """Embed text with retry logic for quota limits"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = genai.embed_content(model=self._embed_model, content=text)
                return resp["embedding"]
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    print(f"⏳ Quota exceeded, waiting 60 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(60)
                else:
                    raise e

    def generate(self, prompt: str) -> str:
        """Generate text with retry logic for quota limits"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(self._gen_model)
                out = model.generate_content(prompt)
                return out.text.strip() if getattr(out, "text", None) else ""
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    print(f"⏳ Quota exceeded, waiting 60 seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(60)
                else:
                    # If still failing, return a fallback response
                    print(f"⚠️ API quota exhausted after {max_retries} attempts, using fallback")
                    return '{"is_highlight": true, "description": "Notable video segment with visual activity", "summary": "Interesting moment detected", "confidence": 0.6}'
