import google.generativeai as genai
from app.config import Config


class GeminiClient:
    def __init__(self):
        if not Config.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing in environment or .env")
        genai.configure(api_key=Config.google_api_key)
        self._embed_model = Config.embedding_model
        self._gen_model = Config.generation_model

    def embed(self, text: str) -> list[float]:
        resp = genai.embed_content(model=self._embed_model, content=text)
        return resp["embedding"]

    def generate(self, prompt: str) -> str:
        model = genai.GenerativeModel(self._gen_model)
        out = model.generate_content(prompt)
        return out.text.strip() if getattr(out, "text", None) else ""
