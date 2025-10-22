from typing import List
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from .models import Base, Video, Highlight
from app.config import Config
from app.types import HighlightModel, VideoRecord
from pgvector.sqlalchemy import Vector


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
            # Use SQLAlchemy ORM with pgvector
            results = s.query(
                Highlight.id,
                Highlight.video_id,
                Highlight.ts_start_sec,
                Highlight.ts_end_sec,
                Highlight.description,
                Highlight.llm_summary,
                Highlight.objects,
                (1 - Highlight.embedding.cosine_distance(query_emb)).label('score')
            ).order_by(
                Highlight.embedding.cosine_distance(query_emb)
            ).limit(top_k).all()
            
            return [
                {
                    "id": r.id,
                    "video_id": r.video_id,
                    "ts_start_sec": r.ts_start_sec,
                    "ts_end_sec": r.ts_end_sec,
                    "description": r.description,
                    "llm_summary": r.llm_summary,
                    "objects": r.objects,
                    "score": float(r.score)
                }
                for r in results
            ]

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword search using ILIKE on description and llm_summary fields"""
        with self.Session() as s:
            # Extract meaningful keywords from the query
            import re
            # Remove common stop words and extract meaningful terms
            stop_words = {'what', 'happened', 'during', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'show', 'me', 'find', 'scenes'}
            words = re.findall(r'\b\w+\b', query.lower())
            keywords = [w for w in words if w not in stop_words and len(w) > 2]
            
            if not keywords:
                # If no meaningful keywords, use the original query
                keywords = [query]
            
            # Build OR conditions for each keyword
            conditions = []
            params = {"k": top_k}
            for i, keyword in enumerate(keywords[:3]):  # Limit to 3 keywords
                param_name = f"pattern{i}"
                conditions.append(f"(description ILIKE :{param_name} OR llm_summary ILIKE :{param_name})")
                params[param_name] = f"%{keyword}%"
            
            where_clause = " OR ".join(conditions)
            
            res = s.execute(
                text(
                    f"""
                    SELECT id, video_id, ts_start_sec, ts_end_sec, description, llm_summary, objects,
                           0.5 AS score
                    FROM highlights
                    WHERE {where_clause}
                    ORDER BY video_id, ts_start_sec
                    LIMIT :k
                    """
                ),
                params,
            )
            return [dict(r._mapping) for r in res]
