from rag_app.chunker import chunk_pages
from rag_app.models import PageText, TextChunk


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


def test_chunk_pages_uses_exact_overlap_slices():
    pages = [PageText("manual", "说明书", 1, "abcdefghij")]

    chunks = chunk_pages(pages, chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_pages_rejects_negative_overlap():
    pages = [PageText("manual", "说明书", 1, "内容")]

    try:
        chunk_pages(pages, chunk_size=20, overlap=-1)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_chunk_pages_rejects_non_positive_chunk_size():
    pages = [PageText("manual", "说明书", 1, "内容")]

    try:
        chunk_pages(pages, chunk_size=0, overlap=0)
    except ValueError as exc:
        assert str(exc) == "chunk_size must be positive"
    else:
        raise AssertionError("Expected ValueError")
