from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rag_app import openai_client
from rag_app.config import Settings


def make_settings(tmp_path, openai_api_key="test-key"):
    return Settings(
        root_dir=tmp_path,
        docs_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        openai_api_key=openai_api_key,
        openai_base_url="https://answer.example/v1",
        openai_model="test-answer-model",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="test-embedding-model",
    )


class FakeEmbeddingsResource:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index), float(index + 1)])
                for index, _ in enumerate(input)
            ]
        )


class FakeChatCompletionsResource:
    def __init__(self):
        self.calls = []
        self.fail_on_images_once = False

    def create(self, model, messages):
        self.calls.append({"model": model, "messages": messages})
        if self.fail_on_images_once and isinstance(messages[1]["content"], list):
            self.fail_on_images_once = False
            raise RuntimeError("No endpoints found that support image input")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="测试回答"))]
        )


class FakeChatResource:
    def __init__(self):
        self.completions = FakeChatCompletionsResource()


class FakeOpenAISDK:
    instances = []

    def __init__(self, api_key, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = FakeEmbeddingsResource()
        self.chat = FakeChatResource()
        self.__class__.instances.append(self)


@pytest.fixture
def fake_openai_sdk(monkeypatch):
    FakeOpenAISDK.instances = []
    monkeypatch.setattr(openai_client, "OpenAI", FakeOpenAISDK)
    return FakeOpenAISDK


def test_missing_answer_api_key_raises_value_error(tmp_path):
    settings = make_settings(tmp_path, openai_api_key=None)
    client = openai_client.OpenAIClient(settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        client.answer_with_context("问题", ["片段"])


def test_whitespace_only_api_key_raises_value_error(tmp_path, fake_openai_sdk):
    settings = make_settings(tmp_path, openai_api_key="   ")
    client = openai_client.OpenAIClient(settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        client.answer_with_context("问题", ["片段"])

    assert [instance.api_key for instance in fake_openai_sdk.instances] == [
        "embedding-key"
    ]


def test_missing_embedding_api_key_raises_value_error(tmp_path, fake_openai_sdk):
    base = make_settings(tmp_path, openai_api_key="answer-key")
    settings = Settings(
        root_dir=base.root_dir,
        docs_dir=base.docs_dir,
        index_dir=base.index_dir,
        openai_api_key=base.openai_api_key,
        openai_base_url=base.openai_base_url,
        openai_model=base.openai_model,
        embedding_api_key=None,
        embedding_base_url=base.embedding_base_url,
        embedding_model=base.embedding_model,
    )
    client = openai_client.OpenAIClient(settings)

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY is not configured"):
        client.embed_texts(["a"])


def test_embed_texts_calls_sdk_with_embedding_model_and_returns_embeddings(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)

    embeddings = client.embed_texts(["a", "b"])

    embedding_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://embedding.example/v1"
    )
    assert embedding_client.api_key == "embedding-key"
    assert str(embedding_client.base_url).rstrip("/") == "https://embedding.example/v1"
    assert embedding_client.embeddings.calls == [
        {"model": settings.embedding_model, "input": ["a", "b"]}
    ]
    assert embeddings == [[0.0, 1.0], [1.0, 2.0]]


def test_embed_texts_splits_requests_to_provider_batch_limit(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)
    texts = [f"text-{index}" for index in range(23)]

    embeddings = client.embed_texts(texts)

    embedding_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://embedding.example/v1"
    )
    assert [len(call["input"]) for call in embedding_client.embeddings.calls] == [
        10,
        10,
        3,
    ]
    assert [call["model"] for call in embedding_client.embeddings.calls] == [
        settings.embedding_model,
        settings.embedding_model,
        settings.embedding_model,
    ]
    assert embeddings == [
        [float(index), float(index + 1)]
        for index in list(range(10)) + list(range(10)) + list(range(3))
    ]


def test_embed_texts_uses_dashscope_multimodal_endpoint_for_vision_embedding_model(
    tmp_path, fake_openai_sdk, monkeypatch
):
    base = make_settings(tmp_path)
    settings = Settings(
        root_dir=base.root_dir,
        docs_dir=base.docs_dir,
        index_dir=base.index_dir,
        openai_api_key=base.openai_api_key,
        openai_base_url=base.openai_base_url,
        openai_model=base.openai_model,
        embedding_api_key=base.embedding_api_key,
        embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_model="tongyi-embedding-vision-flash-2026-03-06",
    )
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "output": {
                        "embeddings": [
                            {"embedding": [0.1, 0.2], "type": "text"},
                            {"embedding": [0.3, 0.4], "type": "text"},
                        ]
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(openai_client, "urlopen", fake_urlopen)
    client = openai_client.OpenAIClient(settings)

    embeddings = client.embed_texts(["文本一", "文本二"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert calls == [
        {
            "url": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            "headers": {
                "Authorization": "Bearer embedding-key",
                "Content-type": "application/json",
            },
            "payload": {
                "model": "tongyi-embedding-vision-flash-2026-03-06",
                "input": {"contents": [{"text": "文本一"}, {"text": "文本二"}]},
                "parameters": {"dimension": 768},
            },
            "timeout": 120,
        }
    ]


def test_answer_with_context_sends_attached_images_to_chat_model(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)

    client.answer_with_context(
        "what does this icon mean?",
        ["source excerpt"],
        images=["data:image/png;base64,abc123"],
    )

    answer_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://answer.example/v1"
    )
    messages = answer_client.chat.completions.calls[0]["messages"]
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "text"
    assert "what does this icon mean?" in user_content[0]["text"]
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }


def test_describe_images_sends_indexing_prompt_and_returns_descriptions(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)

    descriptions = client.describe_images(["data:image/png;base64,abc123"])

    answer_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://answer.example/v1"
    )
    call = answer_client.chat.completions.calls[0]
    user_content = call["messages"][1]["content"]
    assert call["model"] == settings.openai_model
    assert "用于RAG检索" in user_content[0]["text"]
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,abc123"},
    }
    assert descriptions == ["测试回答"]


def test_answer_with_context_retries_without_images_when_provider_rejects_images(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)
    answer_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://answer.example/v1"
    )
    answer_client.chat.completions.fail_on_images_once = True

    answer = client.answer_with_context(
        "what does this icon mean?",
        ["source excerpt"],
        images=["data:image/png;base64,abc123"],
    )

    calls = answer_client.chat.completions.calls
    assert len(calls) == 2
    assert isinstance(calls[0]["messages"][1]["content"], list)
    assert isinstance(calls[1]["messages"][1]["content"], str)
    assert answer


def test_answer_with_context_calls_sdk_with_prompt_and_returns_output_text(
    tmp_path, fake_openai_sdk
):
    settings = make_settings(tmp_path)
    client = openai_client.OpenAIClient(settings)

    answer = client.answer_with_context("问题", ["片段1", "片段2"])

    answer_client = next(
        instance
        for instance in fake_openai_sdk.instances
        if str(instance.base_url).rstrip("/") == "https://answer.example/v1"
    )
    assert answer_client.api_key == settings.openai_api_key
    assert str(answer_client.base_url).rstrip("/") == "https://answer.example/v1"
    assert len(answer_client.chat.completions.calls) == 1
    call = answer_client.chat.completions.calls[0]
    assert call["model"] == settings.openai_model
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "问题" in messages[1]["content"]
    assert "片段1" in messages[1]["content"]
    assert "片段2" in messages[1]["content"]
    assert answer == "测试回答"
