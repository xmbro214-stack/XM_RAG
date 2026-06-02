from __future__ import annotations

import re
from collections.abc import Callable

from rag_app.index_store import LocalIndex
from rag_app.models import AnswerResult, SourceCitation


MAX_DISPLAY_SOURCES = 3
SOURCE_SNIPPET_LENGTH = 180
MIN_RERANK_CANDIDATES = 40
MAX_RERANK_CANDIDATES = 80


class RAGService:
    def __init__(
        self,
        index: LocalIndex,
        openai_client,
        top_k: int,
        min_score: float,
        icon_matcher: Callable[[list[str]], SourceCitation | None] | None = None,
    ):
        self._index = index
        self._openai_client = openai_client
        self._top_k = top_k
        self._min_score = min_score
        self._icon_matcher = icon_matcher

    def answer(self, question: str, images: list[str] | None = None) -> AnswerResult:
        attached_images = images or []
        image_descriptions = (
            self._openai_client.describe_images(attached_images)
            if attached_images
            else []
        )
        query_text = _query_text(question, image_descriptions)
        embeddings = self._openai_client.embed_texts([query_text])
        if not embeddings:
            raise ValueError("Embedding client returned no embeddings")
        query_vector = embeddings[0]
        candidates = self._index.search(
            query_vector,
            top_k=_candidate_limit(self._top_k),
        )
        results = _rerank_results(query_text, candidates)[: self._top_k]
        sources = [
            SourceCitation(
                doc_title=result.chunk.doc_title,
                page=result.chunk.page,
                snippet=_shorten_snippet(result.chunk.text),
                score=round(result.score, 4),
            )
            for result in results[:MAX_DISPLAY_SOURCES]
        ]
        if not results or (results[0].score < self._min_score and not attached_images):
            return AnswerResult(answer="说明书中未找到明确信息。", sources=sources)

        icon_match = (
            self._icon_matcher(attached_images)
            if attached_images and self._icon_matcher
            else None
        )
        if icon_match:
            return AnswerResult(answer=icon_match.snippet, sources=[icon_match])

        direct_answer = _answer_known_indicator(
            question,
            [result.chunk.text for result in results],
            image_descriptions=image_descriptions,
            has_images=bool(attached_images),
        )
        if direct_answer:
            return AnswerResult(answer=direct_answer, sources=sources)

        context_blocks = [_context_block(i, result) for i, result in enumerate(results, start=1)]
        answer = self._openai_client.answer_with_context(
            question,
            context_blocks,
            images=_evidence_images(attached_images, results),
        )
        return AnswerResult(answer=answer, sources=sources)


def _shorten_snippet(text: str) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= SOURCE_SNIPPET_LENGTH:
        return snippet
    return snippet[:SOURCE_SNIPPET_LENGTH].rstrip() + "..."


def _rerank_results(query_text: str, results):
    return sorted(
        results,
        key=lambda result: result.score + _lexical_boost(query_text, result.chunk),
        reverse=True,
    )


def _candidate_limit(top_k: int) -> int:
    if top_k <= 0:
        return top_k
    return min(MAX_RERANK_CANDIDATES, max(MIN_RERANK_CANDIDATES, top_k * 8))


def _lexical_boost(query_text: str, chunk) -> float:
    query = query_text.lower()
    title = (chunk.title or "").lower()
    aliases = [alias.lower() for alias in (chunk.aliases or [])]
    normalized = (chunk.normalized_text or chunk.text or "").lower()
    haystack = " ".join(
        part
        for part in (
            title,
            chunk.chapter_path.lower(),
            normalized,
            " ".join(aliases),
        )
        if part
    )
    boost = 0.0

    for term in _query_terms(query):
        if term in title:
            boost += 0.28
        if term in aliases:
            boost += 0.18
        if term in normalized:
            boost += 0.1

    if _looks_like_toc(chunk):
        boost -= 0.35

    for token in re.findall(r"[a-zA-Z]{2,}", query_text):
        token_lower = token.lower()
        if token_lower in title:
            boost += 0.18
        if token_lower in aliases:
            boost += 0.12
        if f"{token_lower} 是什么" in normalized or f"{token_lower}是什么意思" in normalized:
            boost += 0.1

    if ("什么是" in query or "是什么" in query) and any(
        word in normalized for word in ("简称为", "英文全称", "功能介绍")
    ):
        boost += 0.12
    if "局限性" in title and ("什么是" in query or "是什么" in query):
        boost -= 0.08
    if any(term in query for term in ("拔不出来", "无法拔出", "拔不出")):
        if "应急解锁充电枪" in haystack:
            boost += 0.35
        if "应急拉索" in haystack and "充电枪" in haystack:
            boost += 0.2
        if "无法拔出" in haystack or "无法正常解锁" in haystack:
            boost += 0.16
    if ("慢充" in query or "充电口怎么打开" in query) and "按压充电口盖板后边缘" in haystack:
        boost += 0.25
    return boost


def _query_terms(query: str) -> list[str]:
    terms = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    compact = re.sub(r"[\W_]+", "", query)
    for size in (6, 5, 4, 3, 2):
        for start in range(0, max(0, len(compact) - size + 1)):
            token = compact[start : start + size]
            if _is_useful_chinese_term(token):
                terms.add(token)
    if "拔不出来" in query or "拔不出" in query:
        terms.update({"充电枪", "无法拔出", "应急解锁", "应急拉索", "解锁充电枪"})
    if "慢充" in query:
        terms.update({"交流充电", "充电口", "充电口盖板", "打开充电口"})
    return sorted(terms, key=len, reverse=True)


def _is_useful_chinese_term(token: str) -> bool:
    if not re.search(r"[\u4e00-\u9fff]", token):
        return False
    return token not in {
        "怎么",
        "怎么办",
        "什么",
        "如何",
        "出来",
        "不出",
        "打开",
    }


def _looks_like_toc(chunk) -> bool:
    text = f"{chunk.content_type} {chunk.title} {chunk.normalized_text or chunk.text}"
    return (
        chunk.content_type == "toc"
        or "目录" in text
        or ("重要提示" in text and "页" in text and chunk.page <= 10)
    )


def _context_block(index: int, result) -> str:
    chunk = result.chunk
    heading_parts = [
        chunk.doc_title,
        f"第 {chunk.page} 页",
        chunk.chapter_path,
        chunk.title,
    ]
    heading = " | ".join(part for part in heading_parts if part)
    return f"[{index}] {heading}：{chunk.text}"


def _query_text(question: str, image_descriptions: list[str]) -> str:
    clean_descriptions = [
        " ".join(description.split())
        for description in image_descriptions
        if description.strip()
    ]
    if not clean_descriptions:
        return question
    return f"{question}\n\n图像描述：" + "\n".join(clean_descriptions)


def _evidence_images(
    attached_images: list[str],
    results,
) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for image in attached_images:
        if image not in seen:
            images.append(image)
            seen.add(image)
    for result in results:
        image = result.chunk.image_data_url
        if image and image not in seen:
            images.append(image)
            seen.add(image)
    return images


def _answer_known_indicator(
    question: str,
    source_texts: list[str],
    image_descriptions: list[str] | None = None,
    has_images: bool = False,
) -> str | None:
    joined_sources = " ".join(source_texts)
    signal_text = f"{question} {' '.join(image_descriptions or [])}"
    if (
        ("右箭头" in signal_text or "向右箭头" in signal_text or "右转" in signal_text)
        and "右转向指示灯" in joined_sources
    ):
        return "这是右转向指示灯。开启右转向灯时，此灯点亮并伴随转向提示音。"
    if (
        ("左箭头" in signal_text or "向左箭头" in signal_text or "左转" in signal_text)
        and "左转向指示灯" in joined_sources
    ):
        return "这是左转向指示灯。开启左转向灯时，此灯点亮并伴随转向提示音。"
    if (
        has_images
        and "安全气囊指示灯" in joined_sources
        and ("安全气囊" in signal_text or "airbag" in signal_text.lower())
    ):
        return (
            "这是安全气囊指示灯。此灯点亮表示安全气囊存在故障，"
            "请将车辆停放至安全位置并联系 AITO 用户中心。"
        )
    if "主驾" in question and "主驾安全带未系指示灯" in joined_sources:
        return (
            "这是主驾安全带未系指示灯。此灯点亮表示驾驶员安全带未系，"
            "用于提醒驾驶员及时系好安全带。"
        )
    if (
        ("副驾" in question or "前排乘客" in question)
        and "副驾安全带未系指示灯" in joined_sources
    ):
        return (
            "这是副驾安全带未系指示灯。此灯点亮表示前排乘客安全带未系，"
            "用于提醒前排乘客及时系好安全带。"
        )
    if "后排" in question and "后排安全带未系指示灯" in joined_sources:
        return (
            "这是后排安全带未系指示灯。此灯点亮表示后排有乘客安全带未系，"
            "L 表示后排左，M 表示后排中，R 表示后排右。"
        )
    return None
