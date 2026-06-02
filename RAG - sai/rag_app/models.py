from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    title: str
    path: Path
    page_count: int


@dataclass(frozen=True)
class PageText:
    doc_id: str
    doc_title: str
    page: int
    text: str


@dataclass(frozen=True)
class PageImage:
    doc_id: str
    doc_title: str
    page: int
    image_index: int
    data_url: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str
    content_type: str = "text"
    image_data_url: str | None = None
    page_end: int | None = None
    parent_id: str | None = None
    chapter_path: str = ""
    title: str = ""
    normalized_text: str = ""
    keywords: list[str] | None = None
    aliases: list[str] | None = None
    entities: list[str] | None = None
    safety_level: str = "none"
    related_function: str = ""
    related_system: str = ""
    source_refs: list[dict] | None = None
    image_refs: list[str] | None = None
    visual_description: str = ""
    linked_sections: list[str] | None = None
    retrieval_hints: list[str] | None = None
    answer_template: str = ""


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class SourceCitation:
    doc_title: str
    page: int
    snippet: str
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SourceCitation]
