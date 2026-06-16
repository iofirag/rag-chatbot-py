import json
import os
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import uuid4

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams


class RAGService:
    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "rag_memory")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.llm_model = os.getenv("LLM_MODEL", "llama3.2:3b")
        self.client = QdrantClient(url=self.qdrant_url)
        self._collection_ready = False

    def chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        text_len = len(normalized)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start = max(end - overlap, start + 1)

        return chunks

    def _embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Embedding generation failed: no vector returned by Ollama")
        return embedding

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return

        exists = self.client.collection_exists(self.qdrant_collection)
        if not exists:
            self.client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._collection_ready = True

    def upsert_text_chunks(self, chunks: list[str], base_payload: dict[str, Any]) -> int:
        if not chunks:
            return 0

        points: list[PointStruct] = []
        now = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            embedding = self._embed(chunk)
            self._ensure_collection(len(embedding))
            payload = {
                **base_payload,
                "text": chunk,
                "chunk_index": i,
                "created_at": now,
            }
            points.append(PointStruct(id=str(uuid4()), vector=embedding, payload=payload))

        self.client.upsert(collection_name=self.qdrant_collection, points=points)
        return len(points)

    def store_message(self, conversation_id: str, role: str, text: str) -> None:
        payload = {
            "conversation_id": conversation_id,
            "kind": "message",
            "role": role,
        }
        self.upsert_text_chunks([text], payload)

    def search_related(self, conversation_id: str, query: str, limit: int = 6) -> list[dict[str, Any]]:
        query_vector = self._embed(query)
        self._ensure_collection(len(query_vector))

        q_filter = Filter(
            must=[
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
            ]
        )

        hits = self.client.search(
            collection_name=self.qdrant_collection,
            query_vector=query_vector,
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "score": hit.score,
                "text": (hit.payload or {}).get("text", ""),
                "kind": (hit.payload or {}).get("kind", "unknown"),
                "role": (hit.payload or {}).get("role", ""),
                "filename": (hit.payload or {}).get("filename", ""),
            }
            for hit in hits
            if (hit.payload or {}).get("text")
        ]

    def search_related_with_filters(
        self,
        conversation_id: str,
        query: str,
        limit: int = 8,
        file_filters: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.search_related(conversation_id, query, limit=max(limit * 3, 18))
        selected_files = {item.strip() for item in (file_filters or []) if item and item.strip()}
        if not selected_files:
            return candidates[:limit]

        filtered: list[dict[str, Any]] = []
        for item in candidates:
            if item.get("kind") == "file":
                if item.get("filename") in selected_files:
                    filtered.append(item)
            else:
                filtered.append(item)

        return filtered[:limit]

    def list_files(self, conversation_id: str, limit: int = 1000) -> list[str]:
        result, _ = self.client.scroll(
            collection_name=self.qdrant_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                    FieldCondition(key="kind", match=MatchValue(value="file")),
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )

        files = sorted({(point.payload or {}).get("filename", "") for point in result if (point.payload or {}).get("filename")})
        return files

    def get_history(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        result, _ = self.client.scroll(
            collection_name=self.qdrant_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                    FieldCondition(key="kind", match=MatchValue(value="message")),
                ]
            ),
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )

        messages: list[dict[str, Any]] = []
        for point in result:
            payload = point.payload or {}
            if payload.get("role") not in {"user", "assistant"}:
                continue
            messages.append(
                {
                    "role": payload.get("role"),
                    "text": payload.get("text", ""),
                    "created_at": payload.get("created_at", ""),
                }
            )

        messages.sort(key=lambda item: item.get("created_at", ""))
        return messages

    def build_context_items(self, related_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        context_items: list[dict[str, Any]] = []
        for idx, item in enumerate(related_context):
            citation_id = f"S{idx + 1}"
            source_label = item.get("kind", "unknown")
            if item.get("filename"):
                source_label = f"{source_label}:{item['filename']}"

            context_items.append(
                {
                    "citation_id": citation_id,
                    "source_label": source_label,
                    "text": item.get("text", ""),
                    "kind": item.get("kind", "unknown"),
                    "filename": item.get("filename", ""),
                    "score": item.get("score", 0),
                }
            )
        return context_items

    def _build_ollama_messages(
        self,
        conversation_id: str,
        user_message: str,
        related_context: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        context_items = self.build_context_items(related_context)
        context_block = "\n\n".join(
            [f"[{item['citation_id']}] ({item['source_label']})\n{item['text']}" for item in context_items]
        )

        system_prompt = (
            "You are a RAG chatbot. Use the provided context first. "
            "If the context is not enough, say so clearly and provide the best effort answer. "
            "Use source citations like [S1], [S2] when facts come from retrieved context. "
            "Keep answers concise and factual.\n\n"
            f"Conversation ID: {conversation_id}\n\n"
            f"Retrieved Context:\n{context_block if context_block else 'No retrieved context.'}"
        )

        ollama_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in history[-8:]:
            ollama_messages.append({"role": item["role"], "content": item["text"]})
        ollama_messages.append({"role": "user", "content": user_message})
        return ollama_messages, context_items

    def generate_chat_completion(
        self,
        conversation_id: str,
        user_message: str,
        related_context: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> str:
        ollama_messages, _ = self._build_ollama_messages(conversation_id, user_message, related_context, history)

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": ollama_messages,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("LLM completion failed: no content returned by Ollama")
        return content

    def stream_chat_completion(
        self,
        conversation_id: str,
        user_message: str,
        related_context: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> Generator[str, None, str]:
        ollama_messages, context_items = self._build_ollama_messages(conversation_id, user_message, related_context, history)

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": ollama_messages,
                "stream": True,
            },
            timeout=300,
            stream=True,
        )
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            payload = json.loads(line)
            message = payload.get("message", {})
            delta = message.get("content", "")
            if delta:
                full_text += delta
                yield json.dumps({"type": "delta", "token": delta}) + "\n"

            if payload.get("done"):
                break

        yield json.dumps({"type": "done", "answer": full_text, "contexts": context_items}) + "\n"
        return full_text
