# RAG Chatbot (NotebookLM-style)

A full-stack RAG chatbot using:
- React frontend
- Python FastAPI backend
- Qdrant vector database
- Ollama for embeddings + LLM chat
- Docker Compose for orchestration

## Features

- Upload files and index them into Qdrant with chunking + embeddings
- Store user messages and assistant answers in the same vector DB
- RAG chat completion with context retrieval from Qdrant
- Streaming token-by-token assistant responses in the UI
- Per-file metadata filters to scope retrieval to selected uploaded files
- Source citations for retrieved chunks (S1, S2, ...)
- Conversation history endpoint and UI rendering
- Single server URL for chatbot UI (`GET /`)

## API

- `GET /` -> Chatbot UI app
- `POST /api/upload` -> file upload, chunking, embeddings, insert to Qdrant
- `POST /api/chat` -> RAG chat completion
- `POST /api/chat/stream` -> streaming RAG chat completion (NDJSON events)
- `GET /api/history/{conversation_id}` -> conversation history
- `GET /api/files/{conversation_id}` -> uploaded filenames metadata for filters
- `GET /api/health` -> health status

## Project Structure

- `backend/app/main.py` FastAPI routes and app setup
- `backend/app/rag.py` Qdrant + Ollama RAG service layer
- `frontend/src/App.jsx` React chatbot UI
- `docker-compose.yml` service orchestration

## Run

1. Optional: copy `.env.example` to `.env` and adjust model names.
2. Start everything:

```bash
docker compose up --build
```

3. Open:

- Chatbot UI: http://localhost:8000
- Qdrant API: http://localhost:6333
- Ollama API: http://localhost:11434

## Notes

- The first startup downloads Ollama models and can take time.
- Uploaded file chunks are stored with payload kind `file`.
- User/assistant messages are stored with payload kind `message`.
- Retrieval is scoped by `conversation_id`.
