from __future__ import annotations

from rag_app.models import PageText, TextChunk


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        start = 0
        chunk_number = 0
        text = page.text.strip()
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{page.doc_id}-p{page.page}-c{chunk_number}",
                        doc_id=page.doc_id,
                        doc_title=page.doc_title,
                        page=page.page,
                        text=piece,
                    )
                )
                chunk_number += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks
