from __future__ import annotations
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from sqlalchemy import Integer, String, Text, TIMESTAMP, ForeignKey, func
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(1024), nullable=False)
    video_uid: Mapped[Optional[str]] = mapped_column(String(128), unique=True)
    duration_sec: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())

    highlights: Mapped[List["Highlight"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
    )


class Highlight(Base):
    __tablename__ = "highlights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    ts_start_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_end_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    llm_summary: Mapped[Optional[str]] = mapped_column(Text)
    embedding: Mapped[List[float]] = mapped_column(Vector(768), nullable=False)
    objects: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())

    video: Mapped["Video"] = relationship(back_populates="highlights")

    # Optional: if you prefer SQLAlchemy to create these too (you already do it in init_db.sql)
    # __table_args__ = (
    #     Index("highlights_video_ts_idx", "video_id", "ts_start_sec"),
    # )
