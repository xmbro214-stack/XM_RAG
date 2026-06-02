import numpy as np
import pytest

from rag_app.index_store import LocalIndex, build_index_from_existing_chunks
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


def test_index_rejects_one_dimensional_vectors():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]

    with pytest.raises(ValueError, match="2D|two-dimensional"):
        LocalIndex(vectors=np.array([1.0, 0.0]), chunks=chunks)


def test_index_rejects_vector_chunk_count_mismatch():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="same length"):
        LocalIndex(vectors=vectors, chunks=chunks)


def test_search_rejects_query_dimension_mismatch():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0], top_k=1)


def test_search_rejects_negative_top_k():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    with pytest.raises(ValueError, match="top_k"):
        index.search([1.0, 0.0], top_k=-1)


def test_search_with_zero_top_k_returns_empty_list():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    assert index.search([1.0, 0.0], top_k=0) == []


def test_search_with_top_k_greater_than_result_count_returns_available_results():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "内容")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = LocalIndex(vectors=vectors, chunks=chunks)

    results = index.search([1.0, 0.0], top_k=5)

    assert [result.chunk.chunk_id for result in results] == ["c1"]


def test_load_accepts_structured_manual_chunks(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(
        '{"chunk_id":"m6ev_p20_indicator_airbag","doc_name":"问界M6纯电版使用说明书",'
        '"page_start":20,"page_end":20,"chapter_path":"车辆概览 > 仪表显示屏",'
        '"content_type":"indicator_icon","title":"安全气囊指示灯",'
        '"raw_text":"安全气囊指示灯：此灯点亮，表示安全气囊存在故障。",'
        '"normalized_text":"安全气囊指示灯 气囊故障灯 SRS 灯",'
        '"image_refs":["data/index/images/p20_i4.png"]}\n',
        encoding="utf-8",
    )
    np.save(tmp_path / "vectors.npy", np.array([[1.0, 0.0]], dtype=np.float32))

    index = LocalIndex.load(tmp_path)

    chunk = index.chunks[0]
    assert chunk.chunk_id == "m6ev_p20_indicator_airbag"
    assert chunk.doc_title == "问界M6纯电版使用说明书"
    assert chunk.page == 20
    assert chunk.text.startswith("安全气囊指示灯")
    assert chunk.normalized_text == "安全气囊指示灯 气囊故障灯 SRS 灯"
    assert chunk.image_refs == ["data/index/images/p20_i4.png"]


def test_build_index_from_existing_chunks_embeds_normalized_text_without_rewriting_chunks(
    tmp_path,
):
    chunks_jsonl = tmp_path / "chunks.jsonl"
    original = (
        '{"chunk_id":"m6ev_p21_indicator_tpms","doc_name":"问界M6纯电版使用说明书",'
        '"page_start":21,"page_end":21,"chapter_path":"车辆概览 > 仪表显示屏",'
        '"content_type":"indicator_icon","title":"胎压报警指示灯",'
        '"raw_text":"胎压报警指示灯：此灯点亮，表示胎压监控系统存在故障。",'
        '"normalized_text":"胎压报警指示灯 胎压灯 TPMS 轮胎压力异常"}\n'
    )
    chunks_jsonl.write_text(original, encoding="utf-8")

    class FakeClient:
        def __init__(self):
            self.embedded_texts = []

        def embed_texts(self, texts):
            self.embedded_texts = texts
            return [[1.0, 0.0] for _ in texts]

    client = FakeClient()

    index = build_index_from_existing_chunks(tmp_path, client)

    assert (tmp_path / "vectors.npy").exists()
    assert chunks_jsonl.read_text(encoding="utf-8") == original
    assert client.embedded_texts == ["胎压报警指示灯 胎压灯 TPMS 轮胎压力异常"]
    assert index.chunks[0].title == "胎压报警指示灯"
