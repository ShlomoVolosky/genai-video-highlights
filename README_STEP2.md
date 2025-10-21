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

1. **User Input**: User enters question in React frontend
2. **API Request**: Frontend sends POST to `/chat/query` with question
3. **Search Logic**: Backend determines search method:
   - **With LLM API Key**: Uses embeddings → vector search in pgvector
   - **Without API Key**: Falls back to keyword ILIKE search
4. **Database Query**: Searches highlights table for relevant matches
5. **Response Assembly**: Builds coherent answer from DB-only content
6. **Frontend Display**: Shows answer and matching highlights with timestamps

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

### 1. Vector Search (with LLM API key)
- Uses Claude/Gemini/OpenAI embeddings
- Performs cosine similarity search in pgvector
- Returns semantically relevant highlights
- Sorted by relevance score

### 2. Keyword Search (fallback)
- Uses PostgreSQL ILIKE pattern matching
- Searches `description` and `llm_summary` fields
- Returns text-matching highlights
- Sorted by video_id and timestamp

## 🧪 Testing

The system automatically detects available LLM API keys and chooses the appropriate search method:

1. **With API Key**: Vector embeddings + semantic search
2. **Without API Key**: Keyword search fallback
3. **Database Error**: Graceful error handling

## 🛠️ Technical Details

### Database Integration
- Reuses Step 1 PostgreSQL database
- Queries `highlights` table with pgvector extension
- Maintains referential integrity with `videos` table

### LLM Client Support
- **Claude**: Hash-based 768-dimensional embeddings
- **Gemini**: Native text-embedding-004 model
- **OpenAI**: text-embedding-3-small model
- **Fallback**: Keyword search when no API keys available

### Docker Configuration
- Backend: Python FastAPI with uvicorn
- Frontend: Node.js with Vite dev server
- Database: Shared PostgreSQL from Step 1
- Networking: Internal Docker network communication

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

- "What happened after the person got out of the car?"
- "Show me scenes with people walking"
- "Find highlights about vehicles"
- "What objects were detected in the videos?"
- "Show me the most confident highlights"

The system returns timestamped highlights with descriptions and summaries, allowing users to quickly find relevant video moments.
