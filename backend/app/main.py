import os
import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .rag import RAGService


class ChatRequest(BaseModel):
    conversation_id: str = Field(default="default")
    message: str = Field(min_length=1)
    file_filters: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    contexts: list[dict]


def extract_text_from_file(file_name: str, file_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join([(page.extract_text() or "") for page in reader.pages]).strip()

    # Default fallback is UTF-8 decode for text-oriented formats.
    return file_bytes.decode("utf-8", errors="ignore").strip()


app = FastAPI(title="RAG Chatbot API", version="1.0.0")
rag = RAGService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    conversation_id: str = Form(default="default"),
) -> dict:
    total_chunks = 0
    processed: list[dict[str, str | int]] = []

    for upload in files:
        data = await upload.read()
        if not data:
            continue

        try:
            text = extract_text_from_file(upload.filename or "uploaded_file", data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse {upload.filename}: {exc}") from exc

        if not text:
            continue

        chunks = rag.chunk_text(text)
        if not chunks:
            continue

        inserted = rag.upsert_text_chunks(
            chunks,
            {
                "conversation_id": conversation_id,
                "kind": "file",
                "filename": upload.filename or "uploaded_file",
            },
        )
        total_chunks += inserted
        processed.append({"filename": upload.filename or "uploaded_file", "chunks": inserted})

    return {
        "conversation_id": conversation_id,
        "files": processed,
        "total_chunks": total_chunks,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_completion(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")

    try:
        rag.store_message(payload.conversation_id, "user", message)

        contexts = rag.search_related_with_filters(
            payload.conversation_id,
            message,
            limit=8,
            file_filters=payload.file_filters,
        )
        citation_contexts = rag.build_context_items(contexts)
        history = rag.get_history(payload.conversation_id)
        answer = rag.generate_chat_completion(payload.conversation_id, message, contexts, history)

        rag.store_message(payload.conversation_id, "assistant", answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer, conversation_id=payload.conversation_id, contexts=citation_contexts)


@app.post("/api/chat/stream")
def chat_completion_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty")

    try:
        rag.store_message(payload.conversation_id, "user", message)
        contexts = rag.search_related_with_filters(
            payload.conversation_id,
            message,
            limit=8,
            file_filters=payload.file_filters,
        )
        history = rag.get_history(payload.conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def event_stream():
        assistant_answer = ""
        try:
            for line in rag.stream_chat_completion(payload.conversation_id, message, contexts, history):
                try:
                    event = json.loads(line)
                    if event.get("type") == "done":
                        assistant_answer = event.get("answer", "")
                except Exception:
                    pass
                yield line

            if assistant_answer:
                rag.store_message(payload.conversation_id, "assistant", assistant_answer)
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/api/history/{conversation_id}")
def get_history(conversation_id: str) -> JSONResponse:
    try:
        history = rag.get_history(conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"conversation_id": conversation_id, "messages": history})


@app.get("/api/files/{conversation_id}")
def get_uploaded_files(conversation_id: str) -> JSONResponse:
    try:
        files = rag.list_files(conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"conversation_id": conversation_id, "files": files})


@app.get("/api/history")
def get_default_history() -> JSONResponse:
    try:
        history = rag.get_history("default")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"conversation_id": "default", "messages": history})


frontend_dist = Path(os.getenv("FRONTEND_DIST", "/app/frontend_dist"))
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
