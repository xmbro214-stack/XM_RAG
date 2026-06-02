from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    docs_dir: Path
    index_dir: Path
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str
    top_k: int = 6
    min_score: float = 0.2


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", "data/docs"))
    index_dir = Path(os.getenv("RAG_INDEX_DIR", "data/index"))
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir
    if not index_dir.is_absolute():
        index_dir = root / index_dir
    return Settings(
        root_dir=root,
        docs_dir=docs_dir,
        index_dir=index_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openai_model=os.getenv("OPENAI_MODEL", "mimo-v2.5"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v3"),
    )
