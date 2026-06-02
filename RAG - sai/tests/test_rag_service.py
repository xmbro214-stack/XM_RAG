import numpy as np
import pytest

from rag_app.index_store import LocalIndex
from rag_app.models import SourceCitation, TextChunk
from rag_app.rag_service import RAGService


class FakeOpenAIClient:
    def __init__(self, image_descriptions=None):
        self.embedded_texts = []
        self.described_images = []
        self.image_descriptions = image_descriptions or ["红色安全气囊图标"]

    def embed_texts(self, texts):
        self.embedded_texts = texts
        return [[1.0, 0.0] for _ in texts]

    def answer_with_context(self, question, context_blocks, images=None):
        return f"回答：{question}；引用数量：{len(context_blocks)}"

    def describe_images(self, images):
        self.described_images = images
        return self.image_descriptions


class SpyFakeOpenAIClient(FakeOpenAIClient):
    def __init__(self, embeddings=None, image_descriptions=None):
        super().__init__(image_descriptions=image_descriptions)
        self.embeddings = embeddings
        self.answer_calls = []

    def embed_texts(self, texts):
        if self.embeddings is not None:
            return self.embeddings
        return super().embed_texts(texts)

    def answer_with_context(self, question, context_blocks, images=None):
        self.answer_calls.append((question, context_blocks, images))
        return super().answer_with_context(question, context_blocks, images=images)


def test_fake_client_contract_for_service_tests():
    client = FakeOpenAIClient()

    assert client.embed_texts(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert client.answer_with_context("问题", ["片段"]) == "回答：问题；引用数量：1"


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
            score=1.0,
        )
    ]


def test_answer_limits_and_shortens_displayed_sources():
    chunks = [
        TextChunk(
            f"c{index}",
            "doc",
            "manual",
            index,
            f"page {index} " + ("long source text " * 40),
        )
        for index in range(5)
    ]
    vectors = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.8, 0.2]],
        dtype=np.float32,
    )
    index = LocalIndex(vectors, chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=5, min_score=0.1)

    result = service.answer("how to open the charge port?")

    assert len(result.sources) == 3
    assert all(len(source.snippet) <= 183 for source in result.sources)
    assert all(source.snippet.endswith("...") for source in result.sources)
    assert len(client.answer_calls[0][1]) == 5


def test_answer_reranks_exact_acronym_definition_before_related_limitations():
    chunks = [
        TextChunk(
            "limit",
            "doc",
            "说明书",
            302,
            "LOCP 是一项辅助功能，无法应对所有交通、天气、能见度、道路和车辆状况。",
            title="局限性",
            aliases=["LOCP"],
            normalized_text="LOCP 局限性",
        ),
        TextChunk(
            "definition",
            "doc",
            "说明书",
            302,
            "侧向障碍物防碰撞（Lateral Obstacle Collision Prevention，简称为 LOCP）系统利用摄像头等传感器识别周边行驶环境。",
            title="侧向障碍物防碰撞（LOCP）",
            aliases=["LOCP", "侧向障碍物防碰撞", "Lateral Obstacle Collision Prevention"],
            normalized_text="LOCP 是什么 侧向障碍物防碰撞 Lateral Obstacle Collision Prevention",
        ),
    ]
    vectors = np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
    index = LocalIndex(vectors, chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=2, min_score=0.1)

    result = service.answer("什么是locp")

    assert result.sources[0].snippet.startswith("侧向障碍物防碰撞")
    assert client.answer_calls[0][1][0].startswith("[1] 说明书 | 第 302 页")
    assert "侧向障碍物防碰撞（LOCP）" in client.answer_calls[0][1][0]

def test_answer_reranks_chinese_content_candidate_above_toc_match():
    toc_text = "\u91cd\u8981\u63d0\u793a\uff1a\u8bf7\u67e5\u770b\u7b2c5-7\u9875\u3002\u5145\u7535\u548c\u4f9b\u7535\u76ee\u5f55\u3002"
    emergency_text = (
        "\u5e94\u6025\u89e3\u9501\u5145\u7535\u67aa\u3002"
        "\u5982\u679c\u4ea4\u6d41\u5145\u7535\u67aa\u89e3\u9501\u529f\u80fd\u51fa\u73b0\u6545\u969c\u65f6\uff0c"
        "\u53ef\u4ee5\u4f7f\u7528\u5e94\u6025\u62c9\u7d22\u89e3\u9501\u4ea4\u6d41\u5145\u7535\u67aa\u3002"
    )
    chunks = [
        TextChunk(
            "toc",
            "doc",
            "\u95ee\u754cM6\u7eaf\u7535\u7248\u4f7f\u7528\u8bf4\u660e\u4e66",
            5,
            toc_text,
            content_type="toc",
            title="\u76ee\u5f55",
            normalized_text=toc_text,
        ),
        TextChunk(
            "emergency",
            "doc",
            "\u95ee\u754cM6\u7eaf\u7535\u7248\u4f7f\u7528\u8bf4\u660e\u4e66",
            328,
            emergency_text,
            content_type="section_text",
            title="\u5e94\u6025\u89e3\u9501\u5145\u7535\u67aa",
            normalized_text=emergency_text,
        ),
    ]
    vectors = np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
    index = LocalIndex(vectors, chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    result = service.answer("\u5145\u7535\u67aa\u62d4\u4e0d\u51fa\u6765\u600e\u4e48\u529e\uff1f")

    assert result.sources[0].page == 328
    assert "\u5e94\u6025\u89e3\u9501\u5145\u7535\u67aa" in client.answer_calls[0][1][0]


def test_answer_forwards_attached_images_to_answer_client():
    chunks = [TextChunk("c1", "doc", "manual", 20, "driver seatbelt warning")]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    service.answer("what does this icon mean?", images=["data:image/png;base64,abc123"])

    assert client.answer_calls[0][2] == ["data:image/png;base64,abc123"]


def test_answer_uses_attached_image_description_for_retrieval_query():
    chunks = [TextChunk("c1", "doc", "manual", 20, "安全气囊指示灯说明")]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    service.answer("这是什么意思？", images=["data:image/png;base64,abc123"])

    assert client.described_images == ["data:image/png;base64,abc123"]
    assert client.embedded_texts == ["这是什么意思？\n\n图像描述：红色安全气囊图标"]


def test_answer_forwards_retrieved_visual_chunk_images_to_answer_client():
    chunks = [
        TextChunk(
            "c1",
            "doc",
            "manual",
            20,
            "视觉内容：绿色左箭头图标，表示左转向指示灯。",
            content_type="image",
            image_data_url="data:image/png;base64,from-index",
        )
    ]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    service.answer("这个图标是什么意思？")

    assert client.answer_calls[0][2] == ["data:image/png;base64,from-index"]


def test_answer_uses_specific_driver_seatbelt_indicator_match_without_llm():
    chunks = [
        TextChunk(
            "c1",
            "doc",
            "manual",
            20,
            "仪表指示灯 图标 说明 主驾安全带未系指示灯：此灯点亮，表示驾驶员安全带未系，以提醒驾驶员及时系好安全带。",
        )
    ]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    result = service.answer("主驾仪表亮起这个图标，是什么意思?")

    assert "主驾安全带未系指示灯" in result.answer
    assert "驾驶员安全带未系" in result.answer
    assert client.answer_calls == []


def test_answer_prefers_airbag_indicator_for_attached_icon_question():
    chunks = [
        TextChunk(
            "c1",
            "doc",
            "manual",
            20,
            (
                "仪表指示灯 图标 说明 主驾安全带未系指示灯：此灯点亮，表示驾驶员安全带未系。 "
                "安全气囊指示灯：此灯点亮，表示安全气囊存在故障，请将车辆停放至安全位置并联系 AITO 用户中心。"
            ),
        )
    ]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    result = service.answer(
        "主驾仪表盘这个图标亮起，是什么意思",
        images=["data:image/png;base64,icon"],
    )

    assert "安全气囊指示灯" in result.answer
    assert "安全气囊存在故障" in result.answer
    assert client.answer_calls == []


def test_answer_prefers_right_turn_indicator_when_attached_image_is_green_right_arrow():
    chunks = [
        TextChunk(
            "c1",
            "doc",
            "manual",
            24,
            (
                "右转向指示灯：开启右转向灯时，此灯点亮并伴随转向提示音。 "
                "安全气囊指示灯：此灯点亮，表示安全气囊存在故障。"
            ),
        )
    ]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient(image_descriptions=["绿色右箭头图标"])
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    result = service.answer(
        "这个灯亮起是什么意思",
        images=["data:image/png;base64,green-right-arrow"],
    )

    assert "右转向指示灯" in result.answer
    assert "开启右转向灯时" in result.answer
    assert "安全气囊" not in result.answer


def test_answer_uses_icon_matcher_before_text_heuristics():
    chunks = [
        TextChunk(
            "c1",
            "doc",
            "manual",
            20,
            "主驾安全带未系指示灯：此灯点亮，表示驾驶员安全带未系。 安全气囊指示灯：此灯点亮，表示安全气囊存在故障。",
        )
    ]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()

    def match_icon(images):
        assert images == ["data:image/png;base64,icon"]
        return SourceCitation(
            doc_title="manual",
            page=20,
            snippet="此灯闪烁，表示 ADS 功能故障，请谨慎驾驶并尽快联系 AITO 用户中心。",
            score=0.98,
        )

    service = RAGService(
        index=index,
        openai_client=client,
        top_k=1,
        min_score=0.1,
        icon_matcher=match_icon,
    )

    result = service.answer(
        "主驾仪表盘这个图标亮起，是什么意思",
        images=["data:image/png;base64,icon"],
    )

    assert "ADS 功能故障" in result.answer
    assert result.sources[0].score == 0.98
    assert client.answer_calls == []


def test_answer_uses_icon_match_for_air_suspension_overload_indicator():
    chunks = [TextChunk("c1", "doc", "manual", 24, "此灯点亮，表示路面辅助模式为雪地模式。")]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()

    def match_icon(images):
        return SourceCitation(
            doc_title="manual",
            page=20,
            snippet="这是空气悬架维修模式/超载指示灯。此灯点亮表示车辆超载或空气悬架处于维修模式，已影响空气悬架调节功能。",
            score=0.97,
        )

    service = RAGService(
        index=index,
        openai_client=client,
        top_k=1,
        min_score=0.1,
        icon_matcher=match_icon,
    )

    result = service.answer("这是什么意思", images=["data:image/png;base64,icon"])

    assert "空气悬架维修模式/超载指示灯" in result.answer
    assert "雪地模式" not in result.answer
    assert result.sources[0].page == 20


def test_answer_reports_no_clear_information_when_score_is_low_without_calling_answer_client():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "无关内容")]
    index = LocalIndex(np.array([[0.0, 1.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient()
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.9)

    result = service.answer("慢充口怎么打开？")

    assert result.answer == "说明书中未找到明确信息。"
    assert result.sources[0].page == 1
    assert client.answer_calls == []


def test_answer_raises_clear_error_when_embedding_client_returns_no_embeddings():
    chunks = [TextChunk("c1", "doc", "说明书", 1, "慢充口打开方法。")]
    index = LocalIndex(np.array([[1.0, 0.0]], dtype=np.float32), chunks)
    client = SpyFakeOpenAIClient(embeddings=[])
    service = RAGService(index=index, openai_client=client, top_k=1, min_score=0.1)

    with pytest.raises(ValueError, match="Embedding client returned no embeddings"):
        service.answer("慢充口怎么打开？")
