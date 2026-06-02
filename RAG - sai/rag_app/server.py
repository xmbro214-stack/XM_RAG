from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_app.config import Settings, get_settings
from rag_app.icon_matcher import match_manual_icon
from rag_app.index_store import (
    LocalIndex,
    build_index_from_existing_chunks,
    build_index_from_pdf,
)
from rag_app.openai_client import OpenAIClient
from rag_app.rag_service import RAGService


class AskRequest(BaseModel):
    question: str
    images: list[str] = []


def _has_api_key(settings: Settings) -> bool:
    return bool(settings.openai_api_key and settings.openai_api_key.strip())


def _has_embedding_key(settings: Settings) -> bool:
    return bool(settings.embedding_api_key and settings.embedding_api_key.strip())


def _has_index(settings: Settings) -> bool:
    return (settings.index_dir / "vectors.npy").exists() and (
        settings.index_dir / "chunks.jsonl"
    ).exists()


def _external_failure() -> HTTPException:
    return HTTPException(
        status_code=502,
        detail="Backend service failed. Please retry after checking the configured provider.",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    web_dir = active_settings.root_dir / "web"
    app = FastAPI(title="RAG Knowledge QA")

    @app.get("/")
    def home():
        index_file = web_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(
                status_code=404, detail="Web UI not found. Build web/index.html first."
            )
        return FileResponse(index_file)

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/api/status")
    def status():
        docs = (
            [{"name": path.name} for path in active_settings.docs_dir.glob("*.pdf")]
            if active_settings.docs_dir.exists()
            else []
        )
        return {
            "indexed": _has_index(active_settings),
            "documents": docs,
            "has_api_key": _has_api_key(active_settings)
            and _has_embedding_key(active_settings),
        }

    @app.post("/api/index")
    def build_index():
        if not _has_embedding_key(active_settings):
            raise HTTPException(
                status_code=400, detail="EMBEDDING_API_KEY is not configured"
            )
        chunks_path = active_settings.index_dir / "chunks.jsonl"
        if _has_index(active_settings):
            chunk_count = _count_jsonl_lines(chunks_path)
            return {
                "chunks": chunk_count,
                "document": chunks_path.name,
                "reused": True,
            }
        if chunks_path.exists():
            try:
                client = OpenAIClient(active_settings)
                index = build_index_from_existing_chunks(
                    active_settings.index_dir,
                    client,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise _external_failure() from exc
            return {"chunks": len(index.chunks), "document": chunks_path.name}

        if not _has_api_key(active_settings):
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY is required to caption PDF images during indexing",
            )
        pdfs = list(active_settings.docs_dir.glob("*.pdf"))
        if not pdfs:
            raise HTTPException(
                status_code=404, detail="No PDF files found in data/docs"
            )
        try:
            client = OpenAIClient(active_settings)
            index = build_index_from_pdf(pdfs[0], active_settings.index_dir, client)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise _external_failure() from exc
        return {"chunks": len(index.chunks), "document": pdfs[0].name}

    @app.post("/api/ask")
    def ask(payload: AskRequest):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty")
        if not _has_api_key(active_settings):
            raise HTTPException(
                status_code=400, detail="OPENAI_API_KEY is not configured"
            )
        if not _has_embedding_key(active_settings):
            raise HTTPException(
                status_code=400, detail="EMBEDDING_API_KEY is not configured"
            )
        if not _has_index(active_settings):
            raise HTTPException(
                status_code=400,
                detail="Index not found or incomplete. Please build the index first.",
            )
        try:
            index = LocalIndex.load(active_settings.index_dir)
            service = RAGService(
                index=index,
                openai_client=OpenAIClient(active_settings),
                top_k=active_settings.top_k,
                min_score=active_settings.min_score,
                icon_matcher=lambda images: match_manual_icon(
                    images,
                    active_settings.docs_dir,
                ),
            )
            result = service.answer(question, images=payload.images)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise _external_failure() from exc
        return {
            "answer": result.answer,
            "sources": [source.__dict__ for source in result.sources],
        }

    return app


def _count_jsonl_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


app = create_app()
