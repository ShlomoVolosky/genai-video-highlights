from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class DetectedObjectModel(BaseModel):
    name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class HighlightModel(BaseModel):
    ts_start_sec: int = Field(ge=0)
    ts_end_sec: int = Field(ge=1)
    description: str = Field(min_length=3)
    llm_summary: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    objects: List[DetectedObjectModel] = Field(default_factory=list)
    embedding: Optional[List[float]] = None  # set later

    @field_validator("ts_end_sec")
    @classmethod
    def _end_after_start(cls, v: int, info):
        start = info.data.get("ts_start_sec")
        if start is not None and v <= start:
            raise ValueError("ts_end_sec must be > ts_start_sec")
        return v


class VideoRecord(BaseModel):
    id: int
    source: str
    video_uid: Optional[str] = None
    duration_sec: Optional[int] = Field(default=None, ge=0)
