# RAG Knowledge QA Design

Date: 2026-05-26

## Goal

Build a local web RAG application for answering questions from the knowledge base. The first knowledge source is `问界M6纯电版使用说明书.pdf`, but the storage and metadata model should allow more documents later.

The app will use OpenAI API for embeddings and answer generation. The source PDF and vector index stay local. Every answer must include source page numbers and original text snippets.

## Scope

In scope:

- Local web chat interface.
- Built-in indexing for the current PDF.
- Document model that can support more PDFs later.
- Local vector search over extracted chunks.
- OpenAI API calls for embeddings and final answer generation.
- Answer output with required citations: document name, page number, snippet, and relevance score.
- Clear errors for missing API key, missing index, PDF parsing failure, OpenAI request failure, and low-confidence retrieval.

Out of scope for the first version:

- User account system.
- Cloud deployment.
- Streaming responses.
- Upload UI for arbitrary documents.
- OCR for scanned PDFs.
- Heavy vector database integration.

## Recommended Approach

Use a lightweight self-built RAG stack.

Reasons:

- The project currently has one 388-page PDF, so a simple local NumPy index is enough.
- Source page numbers and snippets must be visible, which is easier when metadata is fully controlled locally.
- The implementation stays easy to debug for a demo.
- The structure can later be upgraded to a vector database or document upload flow without changing the chat API shape.

OpenAI official docs describe embeddings as numerical text representations useful for search-like tasks, and the Responses API as the model response endpoint. This design uses embeddings for retrieval and Responses for final answer generation.

## Architecture

The app has three layers:

1. Indexing layer
   - Reads PDF files from `data/docs/`.
   - Extracts text page by page.
   - Splits text into chunks.
   - Stores chunk text, page number, document id, and document title.
   - Calls OpenAI embeddings API for each chunk.
   - Saves vectors and metadata to `data/index/`.

2. QA layer
   - Receives a user question.
   - Embeds the question.
   - Searches local vectors by cosine similarity.
   - Selects the top 4-6 relevant chunks.
   - Builds a grounded prompt with source labels.
   - Calls OpenAI Responses API.
   - Returns answer text plus source metadata.

3. Web layer
   - Serves a single-page chat UI.
   - Shows knowledge base status at the top.
   - Lets the user ask questions.
   - Displays assistant answers in a chat flow.
   - Shows citations directly below each answer.

## Data Model

Document metadata:

- `doc_id`: stable id derived from filename or manifest.
- `title`: display name.
- `path`: local file path.
- `page_count`: number of pages.
- `indexed_at`: index build timestamp.

Chunk metadata:

- `chunk_id`: unique chunk id.
- `doc_id`: parent document id.
- `doc_title`: display title.
- `page`: 1-based page number.
- `text`: chunk text.
- `embedding_index`: vector row position.

Index files:

- `data/index/vectors.npy`: vector matrix.
- `data/index/chunks.jsonl`: one chunk metadata object per line.
- `data/index/manifest.json`: index version, model names, document list, and build time.

## User Interface

Use the selected chat-flow layout:

- Header: knowledge base title and status, for example `问界 M6 纯电版使用说明书 | 388 页 | 索引已就绪`.
- Main chat area: user questions and assistant answers.
- Citation area below each assistant answer:
  - `第 X 页`
  - document title
  - original snippet
  - relevance score
- Input bar: fixed at the bottom, with send button, loading state, and retry on failure.

The first screen should be the usable chat app, not a landing page.

## API Design

`GET /api/status`

Returns whether the knowledge base is indexed and which documents are available.

`POST /api/index`

Builds or rebuilds the local index. For the first version, this can be triggered from the UI or run automatically if no index exists.

`POST /api/ask`

Request:

```json
{
  "question": "慢充口怎么打开？"
}
```

Response:

```json
{
  "answer": "根据说明书，...",
  "sources": [
    {
      "doc_title": "问界 M6 纯电版使用说明书",
      "page": 316,
      "snippet": "原文片段...",
      "score": 0.82
    }
  ]
}
```

## Prompting Rules

The answer generation prompt must require the model to:

- Answer in Chinese.
- Use only the provided source snippets.
- Prefer concise, practical instructions.
- Include safety warnings when the source includes warning, danger, or attention notes.
- Say that the manual does not contain clear information when the retrieved context is insufficient.
- Avoid guessing beyond the manual.

## Error Handling

- Missing `OPENAI_API_KEY`: show a setup message and disable asking until configured.
- Missing index: show `请先构建索引`.
- PDF parse failure: show the filename and parsing error.
- OpenAI request failure: show a retryable error.
- Retrieval below confidence threshold: answer that no clear matching manual content was found and still show the best candidate snippets if available.
- Empty question: keep the send button disabled.

## Testing

Automated tests should cover:

- PDF page extraction returns page numbers.
- Chunking preserves `doc_id`, `doc_title`, and `page`.
- Vector search returns chunks sorted by similarity.
- `/api/ask` returns both `answer` and `sources` when OpenAI calls are mocked.
- Missing index and missing API key return clear errors.

Manual verification should cover:

- Start the web app locally.
- Build the index for the PDF.
- Ask at least one vehicle-manual question.
- Confirm the answer includes visible page citations and original snippets.
- Confirm an unrelated question is not answered with fabricated manual content.

## Implementation Notes

- Prefer Python for the backend and PDF/indexing logic.
- Use a small local dependency set.
- Use `text-embedding-3-small` as the default embedding model for cost-effective retrieval.
- Use a configurable answer model through environment variables.
- Keep model names in configuration rather than hardcoding them throughout the code.
- Do not commit `.env`, generated index files, or visual brainstorming artifacts.

## Open Questions

- Which OpenAI answer model should be used by default.
- Whether indexing should run automatically on first launch or only after the user clicks a button.

Default decisions for the first implementation plan:

- Use a current OpenAI text model configurable via `OPENAI_MODEL`.
- If the index is missing, the UI shows a build-index action instead of auto-indexing silently.
