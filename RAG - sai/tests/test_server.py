from fastapi.testclient import TestClient

from rag_app.config import Settings
from rag_app.models import AnswerResult, SourceCitation
from rag_app.server import create_app


def make_settings(tmp_path, api_key="test-key"):
    return Settings(
        root_dir=tmp_path,
        docs_dir=tmp_path / "data" / "docs",
        index_dir=tmp_path / "data" / "index",
        openai_api_key=api_key,
        openai_base_url="https://answer.example/v1",
        openai_model="test-chat",
        embedding_api_key=api_key,
        embedding_base_url="https://embedding.example/v1",
        embedding_model="test-embedding",
    )


def test_status_returns_index_state_documents_and_api_key_state(tmp_path):
    settings = make_settings(tmp_path, api_key="   ")
    settings.docs_dir.mkdir(parents=True)
    settings.index_dir.mkdir(parents=True)
    (settings.docs_dir / "manual.pdf").write_bytes(b"%PDF")
    (settings.index_dir / "vectors.npy").write_bytes(b"vectors")
    (settings.index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")
    client = TestClient(create_app(settings))

    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["indexed"] is True
    assert body["documents"] == [{"name": "manual.pdf"}]
    assert body["has_api_key"] is False


def test_ask_rejects_empty_question(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path)))

    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 400
    assert "question" in response.json()["detail"].lower()


def test_ask_requires_complete_index(tmp_path):
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "vectors.npy").write_bytes(b"vectors")
    client = TestClient(create_app(settings))

    response = client.post("/api/ask", json={"question": "What is indexed?"})

    assert response.status_code == 400
    assert "index" in response.json()["detail"].lower()


def test_index_with_no_api_key_returns_400(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path, api_key="   ")))

    response = client.post("/api/index")

    assert response.status_code == 400
    assert "embedding_api_key" in response.json()["detail"].lower()


def test_index_requires_answer_api_key_for_pdf_image_captioning(tmp_path):
    base = make_settings(tmp_path)
    settings = Settings(
        root_dir=base.root_dir,
        docs_dir=base.docs_dir,
        index_dir=base.index_dir,
        openai_api_key="   ",
        openai_base_url=base.openai_base_url,
        openai_model=base.openai_model,
        embedding_api_key="embedding-key",
        embedding_base_url=base.embedding_base_url,
        embedding_model=base.embedding_model,
    )
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "manual.pdf").write_bytes(b"%PDF")
    client = TestClient(create_app(settings))

    response = client.post("/api/index")

    assert response.status_code == 400
    assert "openai_api_key" in response.json()["detail"].lower()


def test_index_with_existing_chunks_only_builds_vectors(tmp_path, monkeypatch):
    base = make_settings(tmp_path)
    settings = Settings(
        root_dir=base.root_dir,
        docs_dir=base.docs_dir,
        index_dir=base.index_dir,
        openai_api_key="   ",
        openai_base_url=base.openai_base_url,
        openai_model=base.openai_model,
        embedding_api_key="embedding-key",
        embedding_base_url=base.embedding_base_url,
        embedding_model=base.embedding_model,
    )
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","doc_name":"manual","page_start":1,'
        '"raw_text":"原文","normalized_text":"检索文本"}\n',
        encoding="utf-8",
    )

    class FakeOpenAIClient:
        def __init__(self, settings):
            self.settings = settings

    class FakeIndex:
        chunks = [object()]

    calls = []

    def fake_build_existing(index_dir, client):
        calls.append((index_dir, client.settings.embedding_api_key))
        return FakeIndex()

    monkeypatch.setattr("rag_app.server.OpenAIClient", FakeOpenAIClient)
    monkeypatch.setattr(
        "rag_app.server.build_index_from_existing_chunks", fake_build_existing
    )
    client = TestClient(create_app(settings))

    response = client.post("/api/index")

    assert response.status_code == 200
    assert response.json()["chunks"] == 1
    assert calls == [(settings.index_dir, "embedding-key")]


def test_index_reuses_complete_existing_index_without_rebuilding(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"c1","doc_name":"manual","page_start":1,'
        '"raw_text":"原文","normalized_text":"检索文本"}\n',
        encoding="utf-8",
    )
    (settings.index_dir / "vectors.npy").write_bytes(b"vectors")

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("index should not be rebuilt")

    monkeypatch.setattr("rag_app.server.build_index_from_existing_chunks", fail_rebuild)
    client = TestClient(create_app(settings))

    response = client.post("/api/index")

    assert response.status_code == 200
    assert response.json() == {
        "chunks": 1,
        "document": "chunks.jsonl",
        "reused": True,
    }


def test_index_with_no_pdfs_returns_404_when_api_key_present(tmp_path):
    settings = make_settings(tmp_path)
    settings.docs_dir.mkdir(parents=True)
    client = TestClient(create_app(settings))

    response = client.post("/api/index")

    assert response.status_code == 404
    assert "pdf" in response.json()["detail"].lower()


def test_home_route_without_web_file_returns_non_500(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path)))

    response = client.get("/")

    assert response.status_code == 404
    assert "web" in response.json()["detail"].lower()


def test_ask_returns_answer_shape_with_mocked_dependencies(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "vectors.npy").write_bytes(b"vectors")
    (settings.index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")

    class FakeIndex:
        pass

    class FakeOpenAIClient:
        def __init__(self, settings):
            self.settings = settings

    class FakeRAGService:
        calls = []

        def __init__(self, index, openai_client, top_k, min_score, icon_matcher=None):
            self.index = index
            self.openai_client = openai_client
            self.top_k = top_k
            self.min_score = min_score
            self.icon_matcher = icon_matcher

        def answer(self, question, images=None):
            self.__class__.calls.append({"question": question, "images": images})
            return AnswerResult(
                answer=f"Answer: {question}",
                sources=[
                    SourceCitation(
                        doc_title="manual",
                        page=1,
                        snippet="source text",
                        score=0.9,
                    )
                ],
            )

    monkeypatch.setattr("rag_app.server.LocalIndex.load", lambda index_dir: FakeIndex())
    monkeypatch.setattr("rag_app.server.OpenAIClient", FakeOpenAIClient)
    monkeypatch.setattr("rag_app.server.RAGService", FakeRAGService)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/ask",
        json={"question": "What now?", "images": ["data:image/png;base64,abc123"]},
    )

    assert response.status_code == 200
    assert FakeRAGService.calls == [
        {"question": "What now?", "images": ["data:image/png;base64,abc123"]}
    ]
    assert response.json() == {
        "answer": "Answer: What now?",
        "sources": [
            {
                "doc_title": "manual",
                "page": 1,
                "snippet": "source text",
                "score": 0.9,
            }
        ],
    }


def test_ask_maps_value_error_to_400(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "vectors.npy").write_bytes(b"vectors")
    (settings.index_dir / "chunks.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "rag_app.server.LocalIndex.load",
        lambda index_dir: (_ for _ in ()).throw(ValueError("bad index")),
    )
    client = TestClient(create_app(settings))

    response = client.post("/api/ask", json={"question": "What now?"})

    assert response.status_code == 400
    assert response.json()["detail"] == "bad index"


def test_index_maps_external_failure_to_502(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.docs_dir.mkdir(parents=True)
    (settings.docs_dir / "manual.pdf").write_bytes(b"%PDF")

    class FakeOpenAIClient:
        def __init__(self, settings):
            self.settings = settings

    def fail_build_index(pdf_path, index_dir, client):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("rag_app.server.OpenAIClient", FakeOpenAIClient)
    monkeypatch.setattr("rag_app.server.build_index_from_pdf", fail_build_index)
    client = TestClient(create_app(settings))

    response = client.post("/api/index")

    assert response.status_code == 502
    assert "retry" in response.json()["detail"].lower()
