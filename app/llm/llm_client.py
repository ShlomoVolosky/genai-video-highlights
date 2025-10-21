from app.config import Config
from typing import Protocol


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients to ensure consistent interface"""
    def embed(self, text: str) -> list[float]: ...
    def generate(self, prompt: str) -> str: ...


class UnifiedLLMClient:
    """
    Unified LLM client that automatically selects between Gemini, OpenAI, and Claude
    based on available API keys in environment.
    """
    
    def __init__(self):
        self.client = self._create_client()
        self.client_type = self._get_client_type()
        print(f"🤖 Using {self.client_type} LLM client")

    def _create_client(self) -> LLMClientProtocol:
        """Create appropriate LLM client based on available API keys"""
        
        # Priority 1: Try Gemini if API key is available
        if Config.google_api_key:
            try:
                from app.llm.gemini_client import GeminiClient
                return GeminiClient()
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini client: {e}")
        
        # Priority 2: Try OpenAI if API key is available
        if Config.openai_api_key:
            try:
                from app.llm.openai_client import OpenAIClient
                return OpenAIClient()
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
        
        # Priority 3: Try Claude if API key is available
        if Config.claude_api_key:
            try:
                from app.llm.claude_client import ClaudeClient
                return ClaudeClient()
            except Exception as e:
                print(f"⚠️ Failed to initialize Claude client: {e}")
        
        # No valid API keys found
        raise RuntimeError(
            "No valid LLM API keys found. Please set one of: GOOGLE_API_KEY, OPENAI_API_KEY, or CLAUDE_API_KEY in your .env file"
        )

    def _get_client_type(self) -> str:
        """Get the type of client being used for logging"""
        if hasattr(self.client, '_gen_model'):
            return "Gemini"
        elif hasattr(self.client, 'client') and hasattr(self.client.client, 'chat'):
            return "OpenAI"
        elif hasattr(self.client, 'client') and hasattr(self.client.client, 'messages'):
            return "Claude"
        else:
            return "Unknown"

    def embed(self, text: str) -> list[float]:
        """Embed text using the active LLM client"""
        return self.client.embed(text)

    def generate(self, prompt: str) -> str:
        """Generate text using the active LLM client"""
        return self.client.generate(prompt)
