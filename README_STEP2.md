# 💬 Step 2: Interactive Chat About Video Highlights

## 🎯 Overview

Step 2 extends the video highlights system with an interactive chat interface that allows users to ask questions about processed video highlights. The system provides answers pulled directly from the database without real-time LLM generation.

## 🏗️ Architecture

### Backend (FastAPI)
```
api/
├── main.py          # FastAPI app with CORS middleware
├── service.py       # ChatService: handles embeddings/keyword search
├── schemas.py       # Pydantic models (ChatQuery, Match, ChatAnswer)
├── routers/
│   └── chat.py      # POST /chat/query endpoint
└── requirements.txt # FastAPI dependencies
```

### Frontend (React + Vite)
```
web/
├── src/
│   ├── main.jsx     # React entry point
│   ├── App.jsx      # Chat UI with input field and results
│   └── api.js       # API client for /chat/query
├── package.json     # React dependencies
├── index.html       # HTML template
└── Dockerfile       # Node.js container
```

## 🔄 Chat Flow

1. **User Input**: User enters natural language question in React frontend
2. **API Request**: Frontend sends POST to `/chat/query` with question
3. **Search Logic**: Backend uses intelligent dual-search approach:
   - **Primary**: Vector search with LLM embeddings (Claude/Gemini/OpenAI)
   - **Fallback**: Smart keyword extraction + PostgreSQL ILIKE search
   - **Auto-fallback**: If vector search returns no results, automatically falls back to keyword search
4. **Natural Language Processing**: Extracts meaningful keywords from questions like "What happened during the journey?" → "journey"
5. **Database Query**: Searches highlights table for semantically or textually relevant matches
6. **Response Assembly**: Builds coherent, timestamped answer from DB-only content
7. **Frontend Display**: Shows structured answer and individual matching highlights

## 🚀 How to Start

### Prerequisites
- Docker and Docker Compose installed
- Step 1 database with processed video highlights
- LLM API key (optional - falls back to keyword search)

### 1. Environment Setup
Ensure your `.env` file contains:
```bash
# Database (reused from Step 1)
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppass
POSTGRES_DB=highlights_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

# LLM API Key (optional - one of these)
CLAUDE_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

### 2. Start Step 1 Database (if not running)
```bash
docker-compose up -d db
```

### 3. Start Step 2 Chat Services
```bash
docker-compose -f docker-compose.chat.yml up --build
```

This starts:
- **chat_api**: FastAPI backend on http://localhost:8000
- **web**: React frontend on http://localhost:5173
- **db**: Reuses existing PostgreSQL database from Step 1

### 4. Access the Chat Interface
Open your browser to: http://localhost:5173

## 📡 API Endpoints

### POST /chat/query
Ask questions about video highlights.

**Request:**
```json
{
  "question": "What happened after the person got out of the car?"
}
```

**Response:**
```json
{
  "answer": "[45s–52s] Person exits vehicle and walks toward building. [67s–74s] Person enters through main entrance.",
  "matches": [
    {
      "id": 123,
      "video_id": 1,
      "ts_start_sec": 45,
      "ts_end_sec": 52,
      "description": "Person exits vehicle",
      "llm_summary": "Person exits vehicle and walks toward building",
      "score": 0.89
    }
  ]
}
```

### GET /docs
FastAPI automatic documentation at http://localhost:8000/docs

## 🔍 Search Methods

### 1. Vector Search (Primary Method)
- **Embeddings**: Uses Claude/Gemini/OpenAI embeddings (768-dimensional vectors)
- **Database**: Performs cosine similarity search in pgvector extension
- **Semantic Understanding**: Finds conceptually related content, not just exact matches
- **Integration**: SQLAlchemy ORM with `cosine_distance()` for optimal performance
- **Sorting**: Results ordered by semantic relevance score

### 2. Smart Keyword Search (Intelligent Fallback)
- **Natural Language Processing**: Extracts meaningful keywords from questions
- **Stop Word Removal**: Filters out common words ("what", "the", "during", etc.)
- **Multi-keyword Search**: Searches up to 3 most relevant keywords with OR logic
- **Database Query**: PostgreSQL ILIKE pattern matching on `description` and `llm_summary`
- **Example**: "What happened during the journey?" → searches for "journey"
- **Sorting**: Results ordered by video_id and timestamp for narrative flow

### 3. Automatic Fallback Logic
- **Primary Attempt**: Always tries vector search first (if LLM client available)
- **Smart Fallback**: If vector search returns 0 results, automatically switches to keyword search
- **Error Handling**: If vector search fails due to technical issues, falls back gracefully
- **No API Key**: Directly uses keyword search when no LLM API keys are configured

## 🧪 Testing

The system automatically detects available LLM API keys and chooses the appropriate search method:

1. **With API Key**: Vector embeddings + semantic search
2. **Without API Key**: Keyword search fallback
3. **Database Error**: Graceful error handling

## 🛠️ Technical Details

### Database Integration
- **PostgreSQL + pgvector**: Reuses Step 1 database with vector extension
- **Vector Storage**: 768-dimensional embeddings stored as `vector(768)` type
- **Indexes**: Optimized with IVFFlat index for fast cosine similarity search
- **Referential Integrity**: Maintains relationships with `videos` table

### LLM Client Support
- **UnifiedLLMClient**: Automatically detects and uses available API keys
- **Claude**: Hash-based 768-dimensional embeddings (deterministic)
- **Gemini**: Native text-embedding-004 model (semantic)
- **OpenAI**: text-embedding-3-small model (768-dim semantic)
- **Priority Order**: Gemini → OpenAI → Claude → Keyword fallback

### Vector Search Implementation
- **SQLAlchemy ORM**: Uses `Highlight.embedding.cosine_distance()` for optimal performance
- **Cosine Similarity**: `1 - cosine_distance` for relevance scoring
- **Error Handling**: Graceful fallback to keyword search on any vector search failure
- **Performance**: Leverages pgvector's optimized C implementation

### Smart Keyword Processing
- **NLP Pipeline**: Extracts meaningful terms from natural language
- **Stop Words**: Removes 50+ common English stop words
- **Multi-term Search**: Combines up to 3 keywords with OR logic
- **Pattern Matching**: Uses PostgreSQL ILIKE for case-insensitive search
- **Field Coverage**: Searches both `description` and `llm_summary` columns

### Docker Configuration
- **Backend**: Python FastAPI with uvicorn auto-reload
- **Frontend**: Node.js with Vite dev server (hot reload)
- **Database**: Shared PostgreSQL from Step 1 with health checks
- **Networking**: Internal Docker network communication
- **Volumes**: Persistent database storage with proper initialization

## 🔧 Development

### Backend Development
```bash
cd api/
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd web/
npm install
npm run dev
```

### Environment Variables
- `VITE_API_BASE`: Frontend API base URL (default: http://localhost:8000)
- `POSTGRES_*`: Database connection parameters
- `*_API_KEY`: LLM service API keys

## 📋 Features

✅ **React Frontend**: Clean, responsive chat interface  
✅ **FastAPI Backend**: RESTful API with automatic documentation  
✅ **Dual Search**: Vector embeddings + keyword fallback  
✅ **Database Integration**: Direct queries to Step 1 highlights  
✅ **Docker Support**: Containerized deployment  
✅ **Error Handling**: Graceful fallbacks and error messages  
✅ **CORS Enabled**: Cross-origin requests supported  
✅ **Real-time**: Instant search results from database  

## 🎯 Example Queries

### ✅ **Working Natural Language Questions:**
- **"What happened during the journey?"** - Returns travel/exploration scenes
- **"Show me dramatic moments"** - Finds fire, action, and high-intensity scenes
- **"Find scenes with movement"** - Locates travel, motion, and dynamic content
- **"What interesting things happened?"** - Returns high-confidence highlights

### ✅ **Single Word Searches:**
- **"journey"** - Travel and exploration content
- **"fire"** - Dramatic scenes with fire and smoke
- **"child"** - Scenic landscapes with children
- **"water"** - Scenes featuring water elements
- **"bicycle"** - Content with bicycles and vehicles

### 🔧 **How It Works:**
1. **Natural Language**: "What happened during the journey?" 
   - Extracts: "journey" 
   - Searches database for journey-related content
   - Returns: 6 timestamped matches about travel scenes

2. **Semantic Understanding**: Even without exact word matches, finds conceptually related content
3. **Coherent Responses**: Combines multiple highlights into narrative flow with timestamps
4. **Database-Only**: All content comes directly from processed video highlights, no AI generation

The system returns timestamped highlights with descriptions and summaries, enabling users to quickly find and understand relevant video moments through natural conversation.
