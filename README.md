# 🎬 GenAI Video Highlights System

A comprehensive AI-powered video processing and chat system that extracts meaningful highlights from videos and enables natural language interaction with the processed content.

## 🏗️ System Overview

This project consists of three main components:

### 📹 **Step 1: Video Processing Pipeline**
Automated video analysis and highlight extraction system that processes videos to identify and extract meaningful moments.

**Key Features:**
- **Multi-format Video Support**: Downloads and processes various video formats
- **Scene Detection**: Identifies scene transitions and visual changes
- **Object Detection**: Recognizes objects, people, and activities in video frames
- **Audio Transcription**: Converts speech to text using Whisper
- **LLM Analysis**: Uses Claude/Gemini/OpenAI to analyze and summarize highlights
- **Vector Embeddings**: Stores semantic embeddings for intelligent search
- **Database Storage**: PostgreSQL with pgvector for efficient storage and retrieval

**Technologies:** Python, OpenCV, FFmpeg, Whisper, PostgreSQL, pgvector, Docker

---

### 💬 **Step 2: Interactive Chat Interface**
Natural language chat system that allows users to ask questions about processed video highlights.

**Key Features:**
- **Natural Language Processing**: Understands complex questions like "What happened during the journey?"
- **Dual Search System**: 
  - Primary: Semantic vector search using LLM embeddings
  - Fallback: Smart keyword extraction with stop-word filtering
- **React Frontend**: Clean, responsive chat interface
- **FastAPI Backend**: RESTful API with automatic documentation
- **Database-Only Responses**: Returns content directly from processed highlights
- **Intelligent Fallback**: Automatically switches search methods for optimal results

**Technologies:** React, FastAPI, SQLAlchemy, pgvector, Docker

**Example Queries:**
- "What happened during the journey?" → Returns travel/exploration scenes
- "Show me dramatic moments" → Finds fire, action, high-intensity scenes
- "Find scenes with movement" → Locates travel, motion, dynamic content

---

### 🧠 **Bonus: Neural Network Tic-Tac-Toe Player**
Self-learning AI that masters Tic-Tac-Toe using reinforcement learning with TensorFlow.

**Key Features:**
- **REINFORCE Algorithm**: Policy gradient learning from scratch
- **Three Difficulty Levels**:
  - Easy: vs Random opponent
  - Medium: vs Heuristic opponent  
  - Hard: Self-play training
- **TensorFlow Implementation**: CPU-only neural network
- **Interactive GUI**: Tkinter interface for human vs AI gameplay
- **Training Visualization**: Loss curves and win/draw/loss rate plots
- **Model Persistence**: Saves trained models for later use

**Technologies:** TensorFlow 2, Keras, NumPy, Matplotlib, Tkinter

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- LLM API key (Claude/Gemini/OpenAI) - optional for Step 2

### Step 1: Video Processing
```bash
# Start database and process videos
docker-compose up --build
```

### Step 2: Interactive Chat
```bash
# Start chat system (includes database, API, and frontend)
docker-compose -f docker-compose.chat.yml up --build

# Access chat interface
open http://localhost:5173
```

### Bonus: Tic-Tac-Toe AI
```bash
# Setup virtual environment
python3 -m venv .venv-ttt && source .venv-ttt/bin/activate
pip install -r requirements-bonus-ttt.txt

# Train AI (choose difficulty: easy/medium/hard)
python bonus/ttt/train.py --opponent medium --episodes 6000

# Play against AI
python bonus/ttt/game.py
```

## 📁 Project Structure

```
video-highlights/
├── app/                     # Step 1: Video processing pipeline
│   ├── main.py             # Main processing script
│   ├── processors/         # Video, audio, scene, object processors
│   ├── llm/               # LLM clients (Claude, Gemini, OpenAI)
│   └── db/                # Database models and repository
├── api/                    # Step 2: FastAPI backend
│   ├── main.py            # FastAPI application
│   ├── service.py         # Chat service with dual search
│   ├── schemas.py         # Pydantic models
│   └── routers/           # API endpoints
├── web/                    # Step 2: React frontend
│   ├── src/               # React components and API client
│   ├── package.json       # Dependencies
│   └── Dockerfile         # Frontend container
├── bonus/                  # Bonus: Tic-Tac-Toe AI
│   └── ttt/               # TensorFlow reinforcement learning
├── tests/                  # Comprehensive test suite
├── docker-compose.yml      # Step 1 services
├── docker-compose.chat.yml # Step 2 services
└── README_*.md            # Detailed documentation
```

## 🔧 Environment Configuration

Create a `.env` file:
```bash
# Database
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppass
POSTGRES_DB=highlights_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5433

# LLM API Keys (choose one or more)
CLAUDE_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Models
EMBEDDING_MODEL=text-embedding-004
GENERATION_MODEL=gemini-1.5-flash
```

## 📚 Documentation

- **[README_STEP1.md](README_STEP1.md)** - Detailed Step 1 video processing documentation
- **[README_STEP2.md](README_STEP2.md)** - Detailed Step 2 chat system documentation  
- **[README_BONUS.md](README_BONUS.md)** - Detailed Bonus tic-tac-toe AI documentation

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_step2_chat.py          # Step 2 chat functionality
pytest tests/test_repository_integration.py  # Database integration
pytest tests/test_llm_and_selector.py   # LLM integration
```

## 🎯 Key Technologies

| Component | Technologies |
|-----------|-------------|
| **Video Processing** | Python, OpenCV, FFmpeg, Whisper, ultralytics |
| **Database** | PostgreSQL, pgvector, SQLAlchemy |
| **LLM Integration** | Claude, Gemini, OpenAI APIs |
| **Backend API** | FastAPI, Pydantic, uvicorn |
| **Frontend** | React, Vite, JavaScript |
| **AI/ML** | TensorFlow 2, Keras, NumPy |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest, pytest-mock |

## ✨ Features Highlights

- 🎬 **Multi-modal Analysis**: Combines video, audio, and visual processing
- 🧠 **Semantic Search**: Vector embeddings for intelligent content discovery
- 💬 **Natural Language**: Chat with your videos using everyday language
- 🔄 **Intelligent Fallback**: Robust search with automatic method switching
- 🐳 **Containerized**: Full Docker support for easy deployment
- 🧪 **Well-Tested**: Comprehensive test suite with mocking
- 🎮 **Bonus AI**: Self-learning game AI with visualization
- 📊 **Scalable**: Modular architecture for easy extension

## 🏆 Project Goals Achieved

✅ **Step 1**: Complete video processing pipeline with LLM analysis  
✅ **Step 2**: Interactive chat system with natural language understanding  
✅ **Bonus**: Self-learning AI with reinforcement learning  
✅ **Integration**: Seamless data flow between all components  
✅ **Documentation**: Comprehensive guides and examples  
✅ **Testing**: Robust test coverage with CI/CD ready structure  

---

**Ready to explore your videos through AI-powered conversation? Start with Step 1 to process your content, then dive into Step 2 for interactive exploration!**