-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: videos
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    source VARCHAR(1024) NOT NULL,
    video_uid VARCHAR(128) UNIQUE,
    duration_sec INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table: highlights
-- Embeddings use 768 dimensions because Gemini's text-embedding-004 model outputs 768-dim vectors.
CREATE TABLE IF NOT EXISTS highlights (
    id SERIAL PRIMARY KEY,
    video_id INT REFERENCES videos(id) ON DELETE CASCADE,
    ts_start_sec INT NOT NULL,
    ts_end_sec INT NOT NULL,
    description TEXT NOT NULL,
    llm_summary TEXT,
    embedding vector(768) NOT NULL,
    objects TEXT,
    confidence NUMERIC,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector index for fast similarity search
-- ivfflat works best with multiple lists; 100 is a reasonable default for small datasets.
CREATE INDEX IF NOT EXISTS highlights_embedding_idx
ON highlights
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Optional additional indexes
CREATE INDEX IF NOT EXISTS highlights_video_ts_idx
ON highlights (video_id, ts_start_sec);
