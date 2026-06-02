# 问界 M6 说明书 RAG 知识问答

这是一个本地运行的 RAG 知识库问答应用。首个知识源是《问界 M6 纯电版使用说明书》，系统会从 PDF 中按页抽取文本、提取内嵌图片、用视觉模型生成图片描述，再统一切分并生成向量索引。回答时会展示来源页码和原文/视觉片段。

当前配置支持两个 OpenAI 兼容接口供应商：

- 回答模型：小米 MiMo `mimo-v2.5`
- 向量模型：阿里百炼 `text-embedding-v3`

## 目录结构

```text
RAG/
├── README.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── rag_app/
│   ├── __init__.py
│   ├── config.py              # 环境变量、路径、模型配置
│   ├── models.py              # 文档、片段、检索结果、引用、回答数据模型
│   ├── pdf_loader.py          # PDF 按页抽取文本和内嵌图片
│   ├── chunker.py             # 文本切分并保留页码/文档元数据
│   ├── index_store.py         # 本地 NumPy 向量索引、保存/加载、索引构建
│   ├── openai_client.py       # OpenAI 兼容接口客户端封装
│   ├── rag_service.py         # RAG 检索、来源构造、回答生成编排
│   └── server.py              # FastAPI Web/API 服务
├── web/
│   ├── index.html             # 本地聊天界面
│   ├── styles.css             # 页面样式
│   └── app.js                 # 前端交互、调用 API、渲染引用来源
├── tests/
│   ├── test_chunker.py
│   ├── test_index_builder.py
│   ├── test_index_store.py
│   ├── test_openai_client.py
│   ├── test_pdf_loader.py
│   ├── test_rag_service.py
│   └── test_server.py
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-05-26-rag-qa-design.md
│       └── plans/
│           └── 2026-05-26-rag-qa-implementation.md
└── data/
    ├── docs/                  # 本地 PDF 文档目录，PDF 不提交
    └── index/                 # 生成的向量索引目录，不提交
```

本地运行时还会出现这些文件或目录，它们已被 `.gitignore` 忽略：

```text
.env                         # 本地密钥和模型配置，不提交
data/docs/*.pdf              # 本地知识库 PDF，不提交
data/index/                  # 构建出的向量索引，不提交
.pytest_cache/
__pycache__/
.superpowers/
.worktrees/
```

## 环境准备

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
OPENAI_API_KEY=小米MiMo的API Key
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
OPENAI_MODEL=mimo-v2.5

EMBEDDING_API_KEY=阿里百炼的API Key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_EMBEDDING_MODEL=text-embedding-v3

RAG_DOCS_DIR=data/docs
RAG_INDEX_DIR=data/index
```

## 放入知识库文档

把 PDF 放到 `data/docs/` 下，例如：

```powershell
Copy-Item ".\问界M6纯电版使用说明书.pdf" ".\data\docs\问界M6纯电版使用说明书.pdf"
```

## 启动应用

```powershell
python -m uvicorn rag_app.server:app --host 127.0.0.1 --port 8000
```

打开浏览器访问：

```text
http://127.0.0.1:8000
```

首次使用时点击页面右上角的“构建索引”。索引完成后即可提问，例如：

```text
慢充口怎么打开？
```

也可以上传或粘贴仪表图标截图后提问：

```text
这个图标亮起是什么意思？
```

回答下方会显示来源文档、页码、相关度和原文/视觉片段。

## API

### `GET /api/status`

返回知识库状态：

```json
{
  "indexed": false,
  "documents": [{"name": "问界M6纯电版使用说明书.pdf"}],
  "has_api_key": true
}
```

### `POST /api/index`

构建或重建本地向量索引。需要 `OPENAI_API_KEY` 和 `EMBEDDING_API_KEY`：前者用于让 `mimo-v2.5` 给 PDF 图片生成检索描述，后者用于生成向量。

成功响应：

```json
{
  "chunks": 1234,
  "document": "问界M6纯电版使用说明书.pdf"
}
```

### `POST /api/ask`

基于知识库提问。需要回答模型 key、向量模型 key，以及已构建索引。

请求：

```json
{
  "question": "慢充口怎么打开？"
}
```

响应：

```json
{
  "answer": "根据说明书...",
  "sources": [
    {
      "doc_title": "问界M6纯电版使用说明书",
      "page": 316,
      "snippet": "原文片段...",
      "score": 0.8123
    }
  ]
}
```

## 测试

运行全部测试：

```powershell
pytest -q
```

当前测试覆盖：

- PDF 页码抽取
- PDF 图片转 PNG data URL 和视觉片段入库
- 文本切分和元数据保留
- 本地向量索引保存、加载和检索
- OpenAI 兼容客户端调用参数
- RAG 服务低置信兜底和引用来源
- FastAPI 状态、索引和问答错误处理

## 注意事项

- `.env` 不要提交到 git。
- `data/index/` 是生成结果，换文档或换 embedding 模型后建议重新构建索引。
- 当前 `/api/index` 默认索引 `data/docs/` 中找到的第一份 PDF。后续如果要支持多文档合并索引，需要扩展索引构建逻辑。
- 如果页面提示 API key 未配置，请检查 `.env` 中 `OPENAI_API_KEY` 和 `EMBEDDING_API_KEY` 是否都有值。
