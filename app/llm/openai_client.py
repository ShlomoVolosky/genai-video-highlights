import openai
import time
from openai import OpenAI
from app.config import Config


class OpenAIClient:
    def __init__(self):
        if not Config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in environment or .env")
        self.client = OpenAI(api_key=Config.openai_api_key)

    def embed(self, text: str) -> list[float]:
        """Embed text using OpenAI's text-embedding-3-small model"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=768  # Match Gemini's 768 dimensions
            )
            return response.data[0].embedding
        except openai.RateLimitError as e:
            print(f"⚠️ OpenAI rate limit exceeded: {e}")
            # For paid accounts, rate limits are usually much higher
            # Just wait a short time and retry once
            time.sleep(5)
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=768
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ OpenAI embedding error: {e}")
            raise e

    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI's GPT model"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model
                messages=[
                    {"role": "system", "content": "You are an expert video analyst. Return only valid JSON as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            print(f"⚠️ OpenAI rate limit exceeded: {e}")
            # For paid accounts, just wait a short time and retry once
            time.sleep(5)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert video analyst. Return only valid JSON as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI generation error: {e}")
            # Fallback for other errors
            return '{"is_highlight": true, "description": "Video segment with detected activity", "summary": "Notable moment", "confidence": 0.5}'
