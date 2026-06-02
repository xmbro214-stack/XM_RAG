from __future__ import annotations

import json
from urllib.request import Request, urlopen

from openai import OpenAI

from rag_app.config import Settings


EMBEDDING_BATCH_SIZE = 10
DASHSCOPE_MULTIMODAL_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)
DASHSCOPE_EMBEDDING_TIMEOUT_SECONDS = 120
DASHSCOPE_VISION_FLASH_DIMENSION = 768


class OpenAIClient:
    def __init__(self, settings: Settings):
        self._answer_client = _make_client(
            settings.openai_api_key,
            settings.openai_base_url,
        )
        self._embedding_client = _make_client(
            settings.embedding_api_key,
            settings.embedding_base_url,
        )
        self._model = settings.openai_model
        self._embedding_model = settings.embedding_model
        self._embedding_api_key = settings.embedding_api_key

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._embedding_api_key or not self._embedding_api_key.strip():
            raise ValueError("EMBEDDING_API_KEY is not configured")
        if _uses_dashscope_multimodal_embedding(self._embedding_model):
            return _embed_texts_with_dashscope_multimodal(
                texts,
                api_key=self._embedding_api_key,
                model=self._embedding_model,
            )

        if self._embedding_client is None:
            raise ValueError("EMBEDDING_API_KEY is not configured")
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = self._embedding_client.embeddings.create(
                model=self._embedding_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def answer_with_context(
        self,
        question: str,
        context_blocks: list[str],
        images: list[str] | None = None,
    ) -> str:
        if self._answer_client is None:
            raise ValueError("OPENAI_API_KEY is not configured")
        context = "\n\n".join(context_blocks)
        user_text = f"Question: {question}\n\nSource excerpts:\n{context}"
        messages = (
            {
                "role": "system",
                "content": (
                    "You are a vehicle manual QA assistant. Answer only from the "
                    "provided source excerpts and any attached images. If an image "
                    "is provided, first identify what is shown, then ground the "
                    "meaning in the closest source excerpt. Use the same language "
                    "as the user. Keep the final answer compact: 1-3 short action "
                    "points, no long source quotations. If the sources do not "
                    'answer clearly, say "说明书中未找到明确信息". Preserve safety '
                    "warnings when relevant."
                ),
            },
            {
                "role": "user",
                "content": _build_user_content(user_text, images or []),
            },
        )
        try:
            response = self._answer_client.chat.completions.create(
                model=self._model,
                messages=list(messages),
            )
        except Exception as exc:
            if images and _is_image_unsupported_error(exc):
                fallback_messages = list(messages)
                fallback_messages[1] = {"role": "user", "content": user_text}
                response = self._answer_client.chat.completions.create(
                    model=self._model,
                    messages=fallback_messages,
                )
            else:
                raise
        return response.choices[0].message.content or ""

    def describe_images(self, images: list[str]) -> list[str]:
        if self._answer_client is None:
            raise ValueError("OPENAI_API_KEY is not configured")

        descriptions: list[str] = []
        for image in images:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You describe vehicle manual images for retrieval. "
                        "Identify icons, colors, labels, table relationships, "
                        "and the likely meaning. Be concise and factual."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_user_content(
                        (
                            "请为这张PDF内的图片生成用于RAG检索的中文描述。"
                            "如果是仪表图标，请描述颜色、形状、文字标识和含义；"
                            "如果无法判断含义，只描述可见内容。"
                        ),
                        [image],
                    ),
                },
            ]
            response = self._answer_client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            descriptions.append(response.choices[0].message.content or "")
        return descriptions


def _build_user_content(text: str, images: list[str]):
    if not images:
        return text

    content = [{"type": "text", "text": text}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image}})
    return content


def _is_image_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "image input" in message or "image_url" in message or "vision" in message


def _uses_dashscope_multimodal_embedding(model: str) -> bool:
    return "embedding-vision" in model or model in {
        "qwen3-vl-embedding",
        "qwen2.5-vl-embedding",
        "multimodal-embedding-v1",
    }


def _embed_texts_with_dashscope_multimodal(
    texts: list[str],
    *,
    api_key: str,
    model: str,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        payload = {
            "model": model,
            "input": {"contents": [{"text": text} for text in batch]},
        }
        if model == "tongyi-embedding-vision-flash-2026-03-06":
            payload["parameters"] = {"dimension": DASHSCOPE_VISION_FLASH_DIMENSION}

        request = Request(
            DASHSCOPE_MULTIMODAL_EMBEDDING_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=DASHSCOPE_EMBEDDING_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        response_embeddings = data.get("output", {}).get("embeddings", [])
        embeddings.extend(item["embedding"] for item in response_embeddings)
    return embeddings


def _make_client(api_key: str | None, base_url: str | None):
    if not api_key or not api_key.strip():
        return None
    kwargs = {"api_key": api_key}
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)
