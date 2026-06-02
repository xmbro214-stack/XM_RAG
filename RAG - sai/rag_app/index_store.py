from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_app.chunker import chunk_pages
from rag_app.models import PageImage, SearchResult, TextChunk
from rag_app.pdf_loader import extract_pdf_images, extract_pdf_pages


class LocalIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[TextChunk]):
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")

        self.chunks = chunks
        if len(self.chunks) != vectors.shape[0]:
            raise ValueError("vectors and chunks must have the same length")
        self.vectors = _normalize(vectors)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        query = np.array(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query vector must be one-dimensional")
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("query vector dimension must match index dimension")

        query = _normalize(query.reshape(1, -1))[0]
        scores = self.vectors @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in order
        ]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: Path) -> "LocalIndex":
        vectors = np.load(index_dir / "vectors.npy")
        chunks = load_chunks(index_dir / "chunks.jsonl")
        return cls(vectors=vectors, chunks=chunks)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def load_chunks(chunks_path: Path) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(_chunk_from_record(json.loads(line)))
    return chunks


def _chunk_from_record(record: dict) -> TextChunk:
    if {"doc_id", "doc_title", "page", "text"}.issubset(record):
        allowed = TextChunk.__dataclass_fields__.keys()
        return TextChunk(**{key: value for key, value in record.items() if key in allowed})

    page = int(record.get("page_start") or record.get("page") or 0)
    raw_text = record.get("raw_text") or record.get("text") or ""
    normalized_text = record.get("normalized_text") or raw_text
    doc_title = record.get("doc_name") or record.get("doc_title") or ""
    return TextChunk(
        chunk_id=record["chunk_id"],
        doc_id=record.get("doc_id") or doc_title,
        doc_title=doc_title,
        page=page,
        text=raw_text,
        content_type=record.get("content_type", "text"),
        image_data_url=record.get("image_data_url"),
        page_end=record.get("page_end"),
        parent_id=record.get("parent_id"),
        chapter_path=record.get("chapter_path", ""),
        title=record.get("title", ""),
        normalized_text=normalized_text,
        keywords=record.get("keywords"),
        aliases=record.get("aliases"),
        entities=record.get("entities"),
        safety_level=record.get("safety_level", "none"),
        related_function=record.get("related_function", ""),
        related_system=record.get("related_system", ""),
        source_refs=record.get("source_refs"),
        image_refs=record.get("image_refs"),
        visual_description=record.get("visual_description", ""),
        linked_sections=record.get("linked_sections"),
        retrieval_hints=record.get("retrieval_hints"),
        answer_template=record.get("answer_template", ""),
    )


def build_index_from_existing_chunks(index_dir: Path, openai_client) -> LocalIndex:
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise ValueError("chunks.jsonl not found")
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks found in chunks.jsonl")
    texts = [chunk.normalized_text or chunk.text for chunk in chunks]
    vectors = np.array(openai_client.embed_texts(texts), dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)
    np.save(index_dir / "vectors.npy", index.vectors)
    return index


def build_index_from_pdf(pdf_path: Path, index_dir: Path, openai_client) -> LocalIndex:
    doc_id = pdf_path.stem
    doc_title = pdf_path.stem
    pages = extract_pdf_pages(pdf_path, doc_id=doc_id, doc_title=doc_title)
    chunks = chunk_pages(pages)
    visual_chunks = _build_visual_chunks(
        extract_pdf_images(pdf_path, doc_id=doc_id, doc_title=doc_title),
        openai_client,
    )
    chunks.extend(visual_chunks)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")
    vectors = np.array(
        openai_client.embed_texts([chunk.text for chunk in chunks]),
        dtype=np.float32,
    )
    index = LocalIndex(vectors=vectors, chunks=chunks)
    index.save(index_dir)
    return index


def _build_visual_chunks(
    images: list[PageImage],
    openai_client,
) -> list[TextChunk]:
    if not images:
        return []

    descriptions = openai_client.describe_images([image.data_url for image in images])
    chunks: list[TextChunk] = []
    for image, description in zip(images, descriptions):
        clean_description = " ".join(description.split())
        if not clean_description:
            continue
        chunks.append(
            TextChunk(
                chunk_id=f"{image.doc_id}-p{image.page}-img{image.image_index}",
                doc_id=image.doc_id,
                doc_title=image.doc_title,
                page=image.page,
                text=(
                    f"视觉内容：{clean_description} "
                    f"来源：第 {image.page} 页第 {image.image_index + 1} 张图片。"
                ),
                content_type="image",
                image_data_url=image.data_url,
            )
        )
    return chunks
