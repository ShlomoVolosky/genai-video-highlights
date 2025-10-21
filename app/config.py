from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    # LLM - Support Gemini, OpenAI, and Claude
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    claude_api_key: str = Field(default="", alias="CLAUDE_API_KEY")

    # DB
    postgres_user: str = Field(default="appuser", alias="POSTGRES_USER")
    postgres_password: str = Field(default="apppass", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="highlights_db", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    # App
    whisper_model: str = Field(default="base", alias="WHISPER_MODEL")
    frame_sample_every_sec: float = Field(default=1.5, alias="FRAME_SAMPLE_EVERY_SEC")
    yolo_model: str = Field(default="yolov8n.pt", alias="YOLO_MODEL")

    embedding_model: str = Field(default="text-embedding-004", alias="EMBEDDING_MODEL")
    generation_model: str = Field(default="gemini-1.5-flash", alias="GENERATION_MODEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("frame_sample_every_sec")
    @classmethod
    def _positive_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("FRAME_SAMPLE_EVERY_SEC must be > 0")
        return v

    def db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Singleton-style settings object importable anywhere
Config = Settings()
