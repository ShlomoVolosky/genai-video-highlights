"""
Tests for Step 2 Chat functionality
"""
import pytest
from unittest.mock import Mock, patch
from api.service import ChatService
from api.schemas import ChatQuery, ChatAnswer, Match


class TestChatService:
    """Test ChatService functionality"""
    
    def test_init_with_claude_api_key(self):
        """Test ChatService initialization with Claude API key"""
        with patch.dict('os.environ', {'CLAUDE_API_KEY': 'test-key'}):
            with patch('api.service.UnifiedLLMClient') as mock_client:
                service = ChatService(top_k=5)
                assert service.top_k == 5
                assert service.repo is not None
                mock_client.assert_called_once()
    
    def test_init_without_api_key(self):
        """Test ChatService initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            service = ChatService(top_k=3)
            assert service.top_k == 3
            assert service.embedder is None
    
    def test_answer_with_embeddings(self):
        """Test answer method with embeddings"""
        mock_repo = Mock()
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_repo.vector_search.return_value = [
            {
                'id': 1,
                'video_id': 1,
                'ts_start_sec': 10,
                'ts_end_sec': 15,
                'description': 'Test description',
                'llm_summary': 'Test summary',
                'score': 0.9
            }
        ]
        
        service = ChatService(top_k=5)
        service.repo = mock_repo
        service.embedder = mock_embedder
        
        answer, matches = service.answer("test question")
        
        mock_embedder.embed.assert_called_once_with("test question")
        mock_repo.vector_search.assert_called_once()
        assert "[10s–15s] Test summary" in answer
        assert len(matches) == 1
    
    def test_answer_with_keyword_search(self):
        """Test answer method with keyword search fallback"""
        mock_repo = Mock()
        mock_repo.keyword_search.return_value = [
            {
                'id': 2,
                'video_id': 1,
                'ts_start_sec': 20,
                'ts_end_sec': 25,
                'description': 'Keyword match',
                'llm_summary': None,
                'score': 0.5
            }
        ]
        
        service = ChatService(top_k=5)
        service.repo = mock_repo
        service.embedder = None  # No embedder available
        
        answer, matches = service.answer("test question")
        
        mock_repo.keyword_search.assert_called_once_with("test question", top_k=5)
        assert "[20s–25s] Keyword match" in answer
        assert len(matches) == 1
    
    def test_answer_no_results(self):
        """Test answer method when no results found"""
        mock_repo = Mock()
        mock_repo.keyword_search.return_value = []
        
        service = ChatService(top_k=5)
        service.repo = mock_repo
        service.embedder = None
        
        answer, matches = service.answer("no results")
        
        assert answer == "I couldn't find relevant highlights for that question."
        assert matches == []
    
    def test_answer_embeddings_fallback_to_keyword(self):
        """Test fallback from embeddings to keyword search on error"""
        mock_repo = Mock()
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = Exception("Embedding failed")
        mock_repo.keyword_search.return_value = [
            {
                'id': 3,
                'video_id': 2,
                'ts_start_sec': 30,
                'ts_end_sec': 35,
                'description': 'Fallback result',
                'llm_summary': 'Fallback summary',
                'score': 0.5
            }
        ]
        
        service = ChatService(top_k=5)
        service.repo = mock_repo
        service.embedder = mock_embedder
        
        answer, matches = service.answer("test question")
        
        mock_embedder.embed.assert_called_once()
        mock_repo.keyword_search.assert_called_once_with("test question", top_k=5)
        assert "[30s–35s] Fallback summary" in answer
        assert len(matches) == 1


class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_chat_query_valid(self):
        """Test valid ChatQuery"""
        query = ChatQuery(question="What happened in the video?")
        assert query.question == "What happened in the video?"
    
    def test_chat_query_too_short(self):
        """Test ChatQuery with too short question"""
        with pytest.raises(ValueError):
            ChatQuery(question="a")
    
    def test_match_schema(self):
        """Test Match schema"""
        match = Match(
            id=1,
            video_id=1,
            ts_start_sec=10,
            ts_end_sec=15,
            description="Test description",
            llm_summary="Test summary",
            score=0.9
        )
        assert match.id == 1
        assert match.video_id == 1
        assert match.ts_start_sec == 10
        assert match.ts_end_sec == 15
        assert match.description == "Test description"
        assert match.llm_summary == "Test summary"
        assert match.score == 0.9
    
    def test_chat_answer_schema(self):
        """Test ChatAnswer schema"""
        matches = [
            Match(
                id=1,
                video_id=1,
                ts_start_sec=10,
                ts_end_sec=15,
                description="Test",
                score=0.9
            )
        ]
        answer = ChatAnswer(
            answer="Test answer",
            matches=matches
        )
        assert answer.answer == "Test answer"
        assert len(answer.matches) == 1
        assert answer.matches[0].id == 1


@pytest.fixture
def mock_repository():
    """Mock repository for testing"""
    with patch('api.service.Repository') as mock:
        yield mock.return_value


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    with patch('api.service.UnifiedLLMClient') as mock:
        yield mock.return_value
