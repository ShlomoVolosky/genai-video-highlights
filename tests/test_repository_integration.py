import os
import pytest
from app.db.repository import Repository
from app.types import HighlightModel, DetectedObjectModel

pytestmark = pytest.mark.integration

def _repo():
    url = os.environ.get("TEST_DB_URL")  # e.g., postgresql+psycopg2://appuser:apppass@localhost:5432/highlights_db
    if not url:
        pytest.skip("TEST_DB_URL not set; skipping integration test")
    return Repository(url)

def test_repo_insert_and_search():
    repo = _repo()
    repo.create_schema()  # no-op if exists

    video = repo.upsert_video("test-source", "VID123", 10)
    assert video.id > 0

    h = HighlightModel(
        ts_start_sec=0,
        ts_end_sec=5,
        description="A person speaking indoors.",
        llm_summary="Person speaks.",
        confidence=0.9,
        objects=[DetectedObjectModel(name="person", confidence=0.95)],
        embedding=[0.01]*768,
    )
    ids = repo.add_highlights(video.id, [h])
    assert len(ids) == 1

    res = repo.vector_search([0.01]*768, top_k=3)
    assert len(res) >= 1
    assert "description" in res[0]
