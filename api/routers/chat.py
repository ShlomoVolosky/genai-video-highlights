from fastapi import APIRouter, HTTPException
from . import __init__  # noqa: F401
from api.schemas import ChatQuery, ChatAnswer, Match
from api.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
service = ChatService(top_k=6)

@router.post("/query", response_model=ChatAnswer)
def query(body: ChatQuery):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    answer, rows = service.answer(q)
    matches = [
        Match(
            id=r["id"],
            video_id=r["video_id"],
            ts_start_sec=r["ts_start_sec"],
            ts_end_sec=r["ts_end_sec"],
            description=r["description"],
            llm_summary=r.get("llm_summary"),
            score=float(r.get("score", 0.0)),
        )
        for r in rows
    ]
    return ChatAnswer(answer=answer, matches=matches)
