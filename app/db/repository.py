from typing import List
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from .models import Base, Video, Highlight
from app.config import Config
from app.types import HighlightModel, VideoRecord


class Repository:
    def __init__(self, url: str | None = None):
        self.engine = create_engine(url or Config.db_url(), echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_schema(self):
        Base.metadata.create_all(self.engine)

    def upsert_video(self, source: str, video_uid: str | None, duration_sec: int | None) -> VideoRecord:
        with self.Session() as s:
            v = None
            if video_uid:
                v = s.execute(select(Video).where(Video.video_uid == video_uid)).scalars().first()
            if not v:
                v = Video(source=source, video_uid=video_uid, duration_sec=duration_sec)
                s.add(v)
            else:
                v.source = source
                v.duration_sec = duration_sec
            s.commit()
            s.refresh(v)
            return VideoRecord(id=v.id, source=v.source, video_uid=v.video_uid, duration_sec=v.duration_sec)

    def add_highlights(self, video_id: int, highlights: List[HighlightModel]) -> List[int]:
        ids: List[int] = []
        with self.Session() as s:
            for h in highlights:
                row = Highlight(
                    video_id=video_id,
                    ts_start_sec=h.ts_start_sec,
                    ts_end_sec=h.ts_end_sec,
                    description=h.description,
                    llm_summary=h.llm_summary,
                    embedding=h.embedding or [],
                    objects=",".join(sorted({o.name for o in h.objects})) or None,
                    confidence=h.confidence,
                )
                s.add(row)
                s.flush()
                ids.append(row.id)
            s.commit()
            return ids

    def vector_search(self, query_emb: list[float], top_k: int = 5) -> list[dict]:
        with self.Session() as s:
            res = s.execute(
                text(
                    """
                    SELECT id, video_id, ts_start_sec, ts_end_sec, description, llm_summary, objects,
                           1 - (embedding <=> :q) AS score
                    FROM highlights
                    ORDER BY embedding <=> :q
                    LIMIT :k
                    """
                ),
                {"q": query_emb, "k": top_k},
            )
            return [dict(r._mapping) for r in res]
