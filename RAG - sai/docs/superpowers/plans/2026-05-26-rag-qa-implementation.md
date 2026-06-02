# RAG Knowledge QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web RAG app that answers questions from `问界M6纯电版使用说明书.pdf` using OpenAI API and always shows page-based source snippets.

**Architecture:** A FastAPI backend serves a static chat UI and JSON APIs. The backend extracts PDF pages, chunks text with page metadata, stores OpenAI embeddings in a local NumPy index, retrieves top chunks for each question, and calls OpenAI Responses API for grounded answers.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, OpenAI Python SDK, pypdf, NumPy, python-dotenv, pytest, vanilla HTML/CSS/JavaScript.

---

## File Structure

- Create `requirements.txt`: runtime and test dependencies.
- Create `.env.example`: documented environment variables.
- Create `data/docs/`: local document folder.
- Create `rag_app/__init__.py`: package marker.
- Create `rag_app/config.py`: environment and path configuration.
- Create `rag_app/models.py`: dataclasses for documents, chunks, search results, and answers.
- Create `rag_app/pdf_loader.py`: PDF page extraction.
- Create `rag_app/chunker.py`: deterministic text chunking.
- Create `rag_app/openai_client.py`: thin OpenAI wrapper for embeddings and answers.
- Create `rag_app/index_store.py`: local vector index build, save, load, and search.
- Create `rag_app/rag_service.py`: retrieval and grounded-answer orchestration.
- Create `rag_app/server.py`: FastAPI routes and static file serving.
- Create `web/index.html`: chat UI.
- Create `web/styles.css`: visual styling.
- Create `web/app.js`: browser logic.
- Create `tests/test_pdf_loader.py`: PDF extraction behavior.
- Create `tests/test_chunker.py`: chunk metadata behavior.
- Create `tests/test_index_store.py`: vector search behavior.
- Create `tests/test_rag_service.py`: mocked RAG answer behavior.
- Create `tests/test_server.py`: API error and shape behavior.
- Modify `.gitignore`: ignore `.env`, virtualenvs, generated indexes, and copied local docs.
- Modify `docs/superpowers/specs/2026-05-26-rag-qa-design.md` only if implementation reveals a spec contradiction.

---

### Task 1: Project Scaffold And Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `data/docs/.gitkeep`
- Create: `rag_app/__init__.py`
- Create: `rag_app/config.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write dependency and environment files**

Create `requirements.txt`:

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
openai>=1.50.0
pypdf>=5.0.0
numpy>=2.0.0
python-dotenv>=1.0.1
pytest>=8.0.0
httpx>=0.27.0
```

Create `.env.example`:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_DOCS_DIR=data/docs
RAG_INDEX_DIR=data/index
```

- [ ] **Step 2: Create package and data markers**

Create empty files:

```text
rag_app/__init__.py
data/docs/.gitkeep
```

- [ ] **Step 3: Write configuration module**

Create `rag_app/config.py`:

```python
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
    openai_model: str
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
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
```

- [ ] **Step 4: Ensure generated artifacts stay out of git**

Update `.gitignore` so it includes:

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc
.superpowers/
data/index/
data/docs/*.pdf
logs/
```

- [ ] **Step 5: Install dependencies**

Run:

```powershell
python -m pip install -r requirements.txt
```

Expected: packages install successfully, including `pypdf`.

- [ ] **Step 6: Verify imports**

Run:

```powershell
python -c "import fastapi, openai, pypdf, numpy, pytest; print('ok')"
```

Expected:

```text
ok
```

- [ ] **Step 7: Commit scaffold**

Run:

```powershell
git add .gitignore .env.example requirements.txt data/docs/.gitkeep rag_app/__init__.py rag_app/config.py
git commit -m "chore: scaffold RAG app"
```

Expected: commit succeeds.

---

### Task 2: Core Models

**Files:**
- Create: `rag_app/models.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write tests for chunk metadata construction**

Create `tests/test_chunker.py` with this initial model test:

```python
from rag_app.models import TextChunk


def test_text_chunk_keeps_source_metadata():
    chunk = TextChunk(
        chunk_id="manual-1-p12-c0",
        doc_id="manual-1",
        doc_title="问界 M6 纯电版使用说明书",
        page=12,
        text="这是一个说明书片段。",
    )

    assert chunk.doc_id == "manual-1"
    assert chunk.page == 12
    assert chunk.text == "这是一个说明书片段。"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/test_chunker.py::test_text_chunk_keeps_source_metadata -v
```

Expected: FAIL with `ModuleNotFoundError` or import error for `rag_app.models`.

- [ ] **Step 3: Implement core dataclasses**

Create `rag_app/models.py`:

```python
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
class TextChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str


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
```

- [ ] **Step 4: Run the model test**

Run:

```powershell
pytest tests/test_chunker.py::test_text_chunk_keeps_source_metadata -v
```

Expected: PASS.

- [ ] **Step 5: Commit models**

Run:

```powershell
git add rag_app/models.py tests/test_chunker.py
git commit -m "feat: add RAG data models"
```

Expected: commit succeeds.

---

### Task 3: PDF Extraction

**Files:**
- Create: `rag_app/pdf_loader.py`
- Test: `tests/test_pdf_loader.py`

- [ ] **Step 1: Write a PDF extraction test using monkeypatched reader**

Create `tests/test_pdf_loader.py`:

```python
from pathlib import Path

from rag_app.pdf_loader import extract_pdf_pages


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, _file):
        self.pages = [FakePage("第一页内容"), FakePage("第二页内容")]


def test_extract_pdf_pages_keeps_one_based_page_numbers(monkeypatch, tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setattr("rag_app.pdf_loader.PdfReader", FakeReader)

    pages = extract_pdf_pages(pdf_path, doc_id="manual", doc_title="说明书")

    assert [page.page for page in pages] == [1, 2]
    assert pages[0].text == "第一页内容"
    assert pages[1].doc_title == "说明书"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/test_pdf_loader.py::test_extract_pdf_pages_keeps_one_based_page_numbers -v
```

Expected: FAIL because `rag_app.pdf_loader` does not exist.

- [ ] **Step 3: Implement PDF extraction**

Create `rag_app/pdf_loader.py`:

```python
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag_app.models import PageText


def extract_pdf_pages(pdf_path: Path, doc_id: str, doc_title: str) -> list[PageText]:
    with pdf_path.open("rb") as handle:
        reader = PdfReader(handle)
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            clean_text = " ".join(text.split())
            if clean_text:
                pages.append(
                    PageText(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=index,
                        text=clean_text,
                    )
                )
        return pages
```

- [ ] **Step 4: Run PDF extraction test**

Run:

```powershell
pytest tests/test_pdf_loader.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit PDF extraction**

Run:

```powershell
git add rag_app/pdf_loader.py tests/test_pdf_loader.py
git commit -m "feat: extract PDF pages"
```

Expected: commit succeeds.

---

### Task 4: Text Chunking

**Files:**
- Create: `rag_app/chunker.py`
- Modify: `tests/test_chunker.py`

- [ ] **Step 1: Add chunking tests**

Append to `tests/test_chunker.py`:

```python
from rag_app.chunker import chunk_pages
from rag_app.models import PageText


def test_chunk_pages_preserves_page_and_doc_metadata():
    pages = [
        PageText(
            doc_id="manual",
            doc_title="说明书",
            page=3,
            text="第一段内容。第二段内容。" * 80,
        )
    ]

    chunks = chunk_pages(pages, chunk_size=80, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.doc_id == "manual" for chunk in chunks)
    assert all(chunk.doc_title == "说明书" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert chunks[0].chunk_id == "manual-p3-c0"


def test_chunk_pages_rejects_overlap_not_smaller_than_chunk_size():
    pages = [PageText("manual", "说明书", 1, "内容")]

    try:
        chunk_pages(pages, chunk_size=20, overlap=20)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_chunker.py -v
```

Expected: FAIL because `rag_app.chunker` does not exist.

- [ ] **Step 3: Implement chunking**

Create `rag_app/chunker.py`:

```python
from __future__ import annotations

from rag_app.models import PageText, TextChunk


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[TextChunk]:
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
```

- [ ] **Step 4: Run chunking tests**

Run:

```powershell
pytest tests/test_chunker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit chunking**

Run:

```powershell
git add rag_app/chunker.py tests/test_chunker.py
git commit -m "feat: chunk extracted pages"
```

Expected: commit succeeds.

---

### Task 5: OpenAI Client Wrapper

**Files:**
- Create: `rag_app/openai_client.py`
- Test: `tests/test_rag_service.py`

- [ ] **Step 1: Write tests around injectable fake client contract**

Create `tests/test_rag_service.py` with this first client-contract test:

```python
class FakeOpenAIClient:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def answer_with_context(self, question, context_blocks):
        return f"回答：{question}；引用数量：{len(context_blocks)}"


def test_fake_client_contract_for_service_tests():
    client = FakeOpenAIClient()

    assert client.embed_texts(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert client.answer_with_context("问题", ["片段"]) == "回答：问题；引用数量：1"
```

- [ ] **Step 2: Run the contract test**

Run:

```powershell
pytest tests/test_rag_service.py::test_fake_client_contract_for_service_tests -v
```

Expected: PASS.

- [ ] **Step 3: Implement OpenAI client wrapper**

Create `rag_app/openai_client.py`:

```python
from __future__ import annotations

from openai import OpenAI

from rag_app.config import Settings


class OpenAIClient:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._embedding_model = settings.embedding_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def answer_with_context(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n".join(context_blocks)
        prompt = (
            "你是车辆说明书知识库助手。请只根据下面的来源片段回答问题。"
            "如果来源没有明确答案，请说“说明书中未找到明确信息”。"
            "回答要简洁、实用，涉及危险、警告、注意事项时必须保留安全提醒。\n\n"
            f"问题：{question}\n\n"
            f"来源片段：\n{context}"
        )
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
        )
        return response.output_text
```

- [ ] **Step 4: Verify module imports without API calls**

Run:

```powershell
python -c "from rag_app.openai_client import OpenAIClient; print(OpenAIClient.__name__)"
```

Expected:

```text
OpenAIClient
```

- [ ] **Step 5: Commit OpenAI wrapper**

Run:

```powershell
git add rag_app/openai_client.py tests/test_rag_service.py
git commit -m "feat: add OpenAI client wrapper"
```

Expected: commit succeeds.

---

### Task 6: Local Vector Index

**Files:**
- Create: `rag_app/index_store.py`
- Create: `tests/test_index_store.py`

- [ ] **Step 1: Write vector search tests**

Create `tests/test_index_store.py`:

```python
import numpy as np

from rag_app.index_store import LocalIndex
from rag_app.models import TextChunk


def test_search_returns_results_sorted_by_cosine_similarity():
    chunks = [
        TextChunk("c1", "doc", "说明书", 1, "充电口说明"),
        TextChunk("c2", "doc", "说明书", 2, "座椅调节说明"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    results = index.search([0.9, 0.1], top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].score > results[1].score


def test_index_round_trip(tmp_path):
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    index.save(tmp_path)
    loaded = LocalIndex.load(tmp_path)

    assert loaded.chunks[0].chunk_id == "c1"
    assert loaded.vectors.shape == (1, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_index_store.py -v
```

Expected: FAIL because `rag_app.index_store` does not exist.

- [ ] **Step 3: Implement local index**

Create `rag_app/index_store.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_app.models import SearchResult, TextChunk


class LocalIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[TextChunk]):
        self.vectors = _normalize(vectors.astype(np.float32))
        self.chunks = chunks
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError("vectors and chunks must have the same length")

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        query = _normalize(np.array([query_vector], dtype=np.float32))[0]
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
        chunks: list[TextChunk] = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                chunks.append(TextChunk(**json.loads(line)))
        return cls(vectors=vectors, chunks=chunks)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
```

- [ ] **Step 4: Run index tests**

Run:

```powershell
pytest tests/test_index_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit local index**

Run:

```powershell
git add rag_app/index_store.py tests/test_index_store.py
git commit -m "feat: add local vector index"
```

Expected: commit succeeds.

---

### Task 7: RAG Service

**Files:**
- Create: `rag_app/rag_service.py`
- Modify: `tests/test_rag_service.py`

- [ ] **Step 1: Add service tests**

Append to `tests/test_rag_service.py`:

```python
import numpy as np

from rag_app.index_store import LocalIndex
from rag_app.models import SourceCitation, TextChunk
from rag_app.rag_service import RAGService


def test_answer_returns_sources_from_retrieved_chunks():
    chunks = [
        TextChunk("c1", "doc", "说明书", 316, "慢充口打开方法。"),
        TextChunk("c2", "doc", "说明书", 20, "仪表显示说明。"),
    ]
    index = LocalIndex(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), chunks)
    client = FakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    result = service.answer("慢充口怎么打开？")

    assert "慢充口怎么打开" in result.answer
    assert result.sources == [
        SourceCitation(
            doc_title="说明书",
            page=316,
            snippet="慢充口打开方法。",
            score=result.sources[0].score,
        )
    ]


def test_answer_reports_no_clear_information_when_score_is_low():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "无关内容")]
    index = LocalIndex(np.array([[0.0, 1.0]], dtype=np.float32), chunks)
    client = FakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.9)

    result = service.answer("慢充口怎么打开？")

    assert result.answer == "说明书中未找到明确信息。"
    assert result.sources[0].page == 1
```

- [ ] **Step 2: Run service tests to verify they fail**

Run:

```powershell
pytest tests/test_rag_service.py -v
```

Expected: FAIL because `rag_app.rag_service` does not exist.

- [ ] **Step 3: Implement RAG service**

Create `rag_app/rag_service.py`:

```python
from __future__ import annotations

from rag_app.index_store import LocalIndex
from rag_app.models import AnswerResult, SourceCitation


class RAGService:
    def __init__(self, index: LocalIndex, openai_client, top_k: int, min_score: float):
        self._index = index
        self._openai_client = openai_client
        self._top_k = top_k
        self._min_score = min_score

    def answer(self, question: str) -> AnswerResult:
        query_vector = self._openai_client.embed_texts([question])[0]
        results = self._index.search(query_vector, top_k=self._top_k)
        sources = [
            SourceCitation(
                doc_title=result.chunk.doc_title,
                page=result.chunk.page,
                snippet=result.chunk.text,
                score=round(result.score, 4),
            )
            for result in results
        ]
        if not results or results[0].score < self._min_score:
            return AnswerResult(answer="说明书中未找到明确信息。", sources=sources)

        context_blocks = [
            f"[{i}] {result.chunk.doc_title} 第 {result.chunk.page} 页：{result.chunk.text}"
            for i, result in enumerate(results, start=1)
        ]
        answer = self._openai_client.answer_with_context(question, context_blocks)
        return AnswerResult(answer=answer, sources=sources)
```

- [ ] **Step 4: Run RAG service tests**

Run:

```powershell
pytest tests/test_rag_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit RAG service**

Run:

```powershell
git add rag_app/rag_service.py tests/test_rag_service.py
git commit -m "feat: add RAG answer service"
```

Expected: commit succeeds.

---

### Task 8: Index Builder

**Files:**
- Modify: `rag_app/index_store.py`
- Create: `tests/test_index_builder.py`

- [ ] **Step 1: Write index builder test**

Create `tests/test_index_builder.py`:

```python
from pathlib import Path

from rag_app.index_store import build_index_from_pdf
from rag_app.models import PageText


class FakeClient:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_build_index_from_pdf_saves_vectors_and_chunks(monkeypatch, tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF fake")
    index_dir = tmp_path / "index"

    def fake_extract(_path: Path, doc_id: str, doc_title: str):
        return [PageText(doc_id, doc_title, 1, "充电说明" * 80)]

    monkeypatch.setattr("rag_app.index_store.extract_pdf_pages", fake_extract)

    index = build_index_from_pdf(pdf, index_dir, FakeClient())

    assert (index_dir / "vectors.npy").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert index.chunks[0].page == 1
```

- [ ] **Step 2: Run builder test to verify it fails**

Run:

```powershell
pytest tests/test_index_builder.py -v
```

Expected: FAIL because `build_index_from_pdf` is not implemented.

- [ ] **Step 3: Implement index builder**

Append to `rag_app/index_store.py`:

```python
from rag_app.chunker import chunk_pages
from rag_app.pdf_loader import extract_pdf_pages


def build_index_from_pdf(pdf_path: Path, index_dir: Path, openai_client) -> LocalIndex:
    doc_id = pdf_path.stem
    doc_title = pdf_path.stem
    pages = extract_pdf_pages(pdf_path, doc_id=doc_id, doc_title=doc_title)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}")
    vectors = np.array(openai_client.embed_texts([chunk.text for chunk in chunks]), dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)
    index.save(index_dir)
    return index
```

- [ ] **Step 4: Run builder test**

Run:

```powershell
pytest tests/test_index_builder.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit index builder**

Run:

```powershell
git add rag_app/index_store.py tests/test_index_builder.py
git commit -m "feat: build index from PDF"
```

Expected: commit succeeds.

---

### Task 9: FastAPI Backend

**Files:**
- Create: `rag_app/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write server tests**

Create `tests/test_server.py`:

```python
from fastapi.testclient import TestClient

from rag_app.server import app


def test_status_returns_index_state():
    client = TestClient(app)

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert "indexed" in body
    assert "documents" in body


def test_ask_rejects_empty_question():
    client = TestClient(app)

    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 400
    assert "question" in response.json()["detail"].lower()
```

- [ ] **Step 2: Run server tests to verify they fail**

Run:

```powershell
pytest tests/test_server.py -v
```

Expected: FAIL because `rag_app.server` does not exist.

- [ ] **Step 3: Implement FastAPI routes**

Create `rag_app/server.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_app.config import get_settings
from rag_app.index_store import LocalIndex, build_index_from_pdf
from rag_app.openai_client import OpenAIClient
from rag_app.rag_service import RAGService


app = FastAPI(title="RAG Knowledge QA")
settings = get_settings()
web_dir = settings.root_dir / "web"


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(web_dir / "index.html")


if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/api/status")
def status():
    indexed = (settings.index_dir / "vectors.npy").exists() and (settings.index_dir / "chunks.jsonl").exists()
    docs = [{"name": path.name} for path in settings.docs_dir.glob("*.pdf")] if settings.docs_dir.exists() else []
    return {
        "indexed": indexed,
        "documents": docs,
        "has_api_key": bool(settings.openai_api_key),
    }


@app.post("/api/index")
def build_index():
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")
    pdfs = list(settings.docs_dir.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail="No PDF files found in data/docs")
    client = OpenAIClient(settings)
    index = build_index_from_pdf(pdfs[0], settings.index_dir, client)
    return {"chunks": len(index.chunks), "document": pdfs[0].name}


@app.post("/api/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")
    if not (settings.index_dir / "vectors.npy").exists():
        raise HTTPException(status_code=400, detail="Index not found. Please build the index first.")
    index = LocalIndex.load(settings.index_dir)
    service = RAGService(
        index=index,
        openai_client=OpenAIClient(settings),
        top_k=settings.top_k,
        min_score=settings.min_score,
    )
    result = service.answer(question)
    return {
        "answer": result.answer,
        "sources": [source.__dict__ for source in result.sources],
    }
```

- [ ] **Step 4: Run server tests**

Run:

```powershell
pytest tests/test_server.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit backend**

Run:

```powershell
git add rag_app/server.py tests/test_server.py
git commit -m "feat: add RAG API server"
```

Expected: commit succeeds.

---

### Task 10: Chat Web UI

**Files:**
- Create: `web/index.html`
- Create: `web/styles.css`
- Create: `web/app.js`

- [ ] **Step 1: Create HTML shell**

Create `web/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>问界 M6 说明书助手</title>
    <link rel="stylesheet" href="/static/styles.css">
  </head>
  <body>
    <main class="app-shell">
      <header class="topbar">
        <div>
          <h1>问界 M6 说明书助手</h1>
          <p id="statusText">正在检查知识库...</p>
        </div>
        <button id="indexButton" type="button">构建索引</button>
      </header>
      <section id="messages" class="messages" aria-live="polite"></section>
      <form id="askForm" class="composer">
        <input id="questionInput" type="text" placeholder="输入说明书问题，例如：慢充口怎么打开？" autocomplete="off">
        <button id="sendButton" type="submit">发送</button>
      </form>
    </main>
    <script src="/static/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Create CSS**

Create `web/styles.css`:

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Microsoft YaHei", Arial, sans-serif;
  background: #f6f7f9;
  color: #20242a;
}

.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto 1fr auto;
}

.topbar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 18px 22px;
  background: #ffffff;
  border-bottom: 1px solid #dde1e7;
}

h1 {
  margin: 0 0 4px;
  font-size: 20px;
}

#statusText {
  margin: 0;
  color: #626c78;
  font-size: 14px;
}

button {
  border: 0;
  border-radius: 6px;
  padding: 10px 14px;
  background: #166534;
  color: #ffffff;
  cursor: pointer;
  font-size: 14px;
}

button:disabled {
  background: #9aa3ad;
  cursor: not-allowed;
}

.messages {
  padding: 24px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message {
  max-width: 880px;
  width: min(880px, 100%);
  padding: 14px 16px;
  border-radius: 8px;
  line-height: 1.65;
}

.message.user {
  align-self: flex-end;
  background: #dbeafe;
}

.message.assistant {
  align-self: flex-start;
  background: #ffffff;
  border: 1px solid #dde1e7;
}

.sources {
  margin-top: 12px;
  display: grid;
  gap: 8px;
}

.source {
  padding: 10px;
  border-radius: 6px;
  background: #f6f7f9;
  border: 1px solid #e2e6ec;
  font-size: 13px;
}

.source strong {
  display: block;
  margin-bottom: 4px;
}

.composer {
  display: flex;
  gap: 10px;
  padding: 16px 22px;
  background: #ffffff;
  border-top: 1px solid #dde1e7;
}

.composer input {
  flex: 1;
  min-width: 0;
  border: 1px solid #cbd2dc;
  border-radius: 6px;
  padding: 12px;
  font-size: 15px;
}
```

- [ ] **Step 3: Create browser logic**

Create `web/app.js`:

```javascript
const statusText = document.querySelector("#statusText");
const indexButton = document.querySelector("#indexButton");
const messages = document.querySelector("#messages");
const askForm = document.querySelector("#askForm");
const questionInput = document.querySelector("#questionInput");
const sendButton = document.querySelector("#sendButton");

async function refreshStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  const docs = status.documents.map((doc) => doc.name).join("，") || "未发现 PDF";
  const state = status.indexed ? "索引已就绪" : "索引未构建";
  const key = status.has_api_key ? "API key 已配置" : "API key 未配置";
  statusText.textContent = `${docs} | ${state} | ${key}`;
}

function addMessage(role, text, sources = []) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const body = document.createElement("div");
  body.textContent = text;
  node.appendChild(body);

  if (sources.length > 0) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    for (const source of sources) {
      const item = document.createElement("div");
      item.className = "source";
      item.innerHTML = `<strong>${source.doc_title} | 第 ${source.page} 页 | 相关度 ${source.score}</strong><span></span>`;
      item.querySelector("span").textContent = source.snippet;
      sourceList.appendChild(item);
    }
    node.appendChild(sourceList);
  }

  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

indexButton.addEventListener("click", async () => {
  indexButton.disabled = true;
  indexButton.textContent = "构建中...";
  try {
    const response = await fetch("/api/index", { method: "POST" });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "构建索引失败");
    }
    const result = await response.json();
    addMessage("assistant", `索引构建完成：${result.document}，共 ${result.chunks} 个片段。`);
    await refreshStatus();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    indexButton.disabled = false;
    indexButton.textContent = "构建索引";
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  addMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "发送中...";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "请求失败");
    }
    const result = await response.json();
    addMessage("assistant", result.answer, result.sources);
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "发送";
    questionInput.focus();
  }
});

refreshStatus();
```

- [ ] **Step 4: Smoke test static files exist**

Run:

```powershell
Test-Path web/index.html; Test-Path web/styles.css; Test-Path web/app.js
```

Expected:

```text
True
True
True
```

- [ ] **Step 5: Commit web UI**

Run:

```powershell
git add web/index.html web/styles.css web/app.js
git commit -m "feat: add chat web UI"
```

Expected: commit succeeds.

---

### Task 11: Local Document Placement And End-To-End Verification

**Files:**
- Modify: none required unless verification reveals a bug.

- [ ] **Step 1: Copy the PDF into the document folder**

Run:

```powershell
Copy-Item -LiteralPath '.\问界M6纯电版使用说明书.pdf' -Destination '.\data\docs\问界M6纯电版使用说明书.pdf'
```

Expected: the copied PDF exists in `data/docs/` and remains ignored by git.

- [ ] **Step 2: Create local environment file**

Create `.env` from `.env.example` and set a real key:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_DOCS_DIR=data/docs
RAG_INDEX_DIR=data/index
```

- [ ] **Step 3: Run all automated tests**

Run:

```powershell
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Start the app**

Run:

```powershell
python -m uvicorn rag_app.server:app --host 127.0.0.1 --port 8000
```

Expected: Uvicorn starts and serves `http://127.0.0.1:8000`.

- [ ] **Step 5: Build the index from the UI**

Open:

```text
http://127.0.0.1:8000
```

Click `构建索引`.

Expected: UI reports index built and `data/index/vectors.npy` plus `data/index/chunks.jsonl` exist.

- [ ] **Step 6: Ask a grounded manual question**

In the UI, ask:

```text
慢充口怎么打开？
```

Expected: answer appears in Chinese and includes sources below the answer with page numbers and original snippets.

- [ ] **Step 7: Ask an unrelated question**

In the UI, ask:

```text
这辆车适合种草莓吗？
```

Expected: answer says the manual does not contain clear information or does not fabricate vehicle-manual facts; best candidate snippets may still display.

- [ ] **Step 8: Commit verification fixes if needed**

If code changes were required during verification, run:

```powershell
git status --short
git add <changed-files>
git commit -m "fix: polish RAG verification issues"
```

Expected: only intentional source or test files are committed.

---

## Self-Review

Spec coverage:

- Local web chat interface: Task 10.
- Built-in current PDF with future multi-document structure: Tasks 1, 8, 11.
- OpenAI embeddings and answer generation: Tasks 5, 7, 8.
- Required page and snippet citations: Tasks 2, 3, 4, 7, 10.
- Local vector index: Task 6.
- Clear errors: Task 9 and Task 10.
- Testing: Tasks 2 through 9 and Task 11.

Placeholder scan:

- No unfinished placeholder markers or unfilled implementation gaps are intentionally left in the plan.

Type consistency:

- `TextChunk`, `SourceCitation`, `AnswerResult`, `LocalIndex`, and `RAGService` names are used consistently across implementation and tests.
