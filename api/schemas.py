from pydantic import BaseModel, Field
from typing import List, Optional

class ChatQuery(BaseModel):
    question: str = Field(min_length=2)

class Match(BaseModel):
    id: int
    video_id: int
    ts_start_sec: int
    ts_end_sec: int
    description: str
    llm_summary: Optional[str] = None
    score: float

class ChatAnswer(BaseModel):
    answer: str
    matches: List[Match]
