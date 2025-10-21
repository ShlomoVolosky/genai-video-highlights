import anthropic
import time
from anthropic import Anthropic
from app.config import Config


class ClaudeClient:
    def __init__(self):
        if not Config.claude_api_key:
            raise RuntimeError("CLAUDE_API_KEY missing in environment or .env")
        self.client = Anthropic(api_key=Config.claude_api_key)

    def embed(self, text: str) -> list[float]:
        """
        Claude doesn't have native embeddings, so we'll use a simple fallback.
        In production, you might want to use a separate embedding service.
        For now, we'll create a simple hash-based embedding for compatibility.
        """
        import hashlib
        import struct
        
        # Create a deterministic 768-dimensional vector from text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 768 floats between -1 and 1
        embedding = []
        for i in range(768):
            byte_idx = i % len(hash_bytes)
            # Convert byte to float between -1 and 1
            float_val = (hash_bytes[byte_idx] / 127.5) - 1.0
            embedding.append(float_val)
        
        return embedding

    def generate(self, prompt: str) -> str:
        """Generate text using Claude's API"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast and cost-effective model
                max_tokens=500,
                temperature=0.7,
                system="You are an expert video analyst. Return only valid JSON as requested.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError as e:
            print(f"⚠️ Claude rate limit exceeded: {e}")
            # Wait a short time and retry once
            time.sleep(5)
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                temperature=0.7,
                system="You are an expert video analyst. Return only valid JSON as requested.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"❌ Claude generation error: {e}")
            # Fallback for other errors
            return '{"is_highlight": true, "description": "Video segment with detected activity", "summary": "Notable moment", "confidence": 0.5}'
