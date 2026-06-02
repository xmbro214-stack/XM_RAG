from pathlib import Path

import pytest

from rag_app.index_store import build_index_from_pdf
from rag_app.models import PageImage, PageText


class FakeClient:
    def __init__(self):
        self.embedded_texts = []
        self.described_images = []

    def embed_texts(self, texts):
        self.embedded_texts = texts
        return [[1.0, 0.0] for _ in texts]

    def describe_images(self, images):
        self.described_images = images
        return ["绿色左箭头图标，表示左转向指示灯。"]


def test_build_index_from_pdf_saves_vectors_and_chunks(monkeypatch, tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF fake")
    index_dir = tmp_path / "index"

    def fake_extract(_path: Path, doc_id: str, doc_title: str):
        assert _path == pdf
        assert doc_id == "manual"
        assert doc_title == "manual"
        return [PageText(doc_id, doc_title, 1, "充电说明" * 80)]

    monkeypatch.setattr("rag_app.index_store.extract_pdf_pages", fake_extract)
    monkeypatch.setattr("rag_app.index_store.extract_pdf_images", lambda *args, **kwargs: [])

    index = build_index_from_pdf(pdf, index_dir, FakeClient())

    assert (index_dir / "vectors.npy").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert index.chunks[0].page == 1


def test_build_index_from_pdf_adds_visual_chunks_from_pdf_images(monkeypatch, tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF fake")
    index_dir = tmp_path / "index"
    client = FakeClient()

    def fake_extract_text(_path: Path, doc_id: str, doc_title: str):
        return [PageText(doc_id, doc_title, 12, "仪表指示灯说明")]

    def fake_extract_images(_path: Path, doc_id: str, doc_title: str):
        return [
            PageImage(
                doc_id=doc_id,
                doc_title=doc_title,
                page=12,
                image_index=0,
                data_url="data:image/png;base64,abc123",
            )
        ]

    monkeypatch.setattr("rag_app.index_store.extract_pdf_pages", fake_extract_text)
    monkeypatch.setattr("rag_app.index_store.extract_pdf_images", fake_extract_images)

    index = build_index_from_pdf(pdf, index_dir, client)

    visual_chunk = next(chunk for chunk in index.chunks if chunk.content_type == "image")
    assert visual_chunk.page == 12
    assert visual_chunk.image_data_url == "data:image/png;base64,abc123"
    assert "绿色左箭头图标" in visual_chunk.text
    assert client.described_images == ["data:image/png;base64,abc123"]
    assert any("绿色左箭头图标" in text for text in client.embedded_texts)


def test_build_index_from_pdf_raises_when_no_chunks(monkeypatch, tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF fake")
    index_dir = tmp_path / "index"

    def fake_extract(_path: Path, doc_id: str, doc_title: str):
        return []

    monkeypatch.setattr("rag_app.index_store.extract_pdf_pages", fake_extract)
    monkeypatch.setattr("rag_app.index_store.extract_pdf_images", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="No extractable text"):
        build_index_from_pdf(pdf, index_dir, FakeClient())
