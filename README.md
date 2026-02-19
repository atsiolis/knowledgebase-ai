# 🧠 KnowledgeBase AI

An AI-powered document Q&A system with a real-time RAG (Retrieval-Augmented Generation) pipeline. Upload PDFs or text files and get instant, accurate answers grounded in your documents — complete with source citations streamed token-by-token.

---

## ✨ Features

- **Document Ingestion** — Upload PDFs or plain text files and have them automatically chunked, embedded, and stored
- **Semantic Search** — Uses OpenAI embeddings and Supabase's pgvector store to find the most relevant passages for any query
- **Streaming Answers** — GPT-4o responses are streamed token-by-token via Server-Sent Events, rendering live in the UI
- **Source Citations** — Every answer references the source documents it was derived from
- **Real-time Upload Progress** — Progress bar tracks every stage from extraction to database insert
- **Clean Frontend** — Intuitive HTML/CSS/JS interface with live markdown rendering

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | Supabase (pgvector) |
| LLM | OpenAI GPT-4o (streaming) |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
knowledgebase-ai/
├── app/
│   ├── db/
│   │   └── supabase_client.py    # Supabase connection & client setup
│   ├── rag/
│   │   ├── ingestion.py          # Document parsing, chunking & batch embedding
│   │   ├── llm_chain.py          # Streaming answer generation via AsyncOpenAI
│   │   └── retriever.py          # Async vector similarity search
│   └── main.py                   # FastAPI entry point, routes & SSE streaming
├── frontend/
│   ├── index.html                # Main UI
│   ├── scripts.js                # SSE stream consumer & UI logic
│   └── style.css                 # Styling + streaming cursor animation
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project
- An [OpenAI](https://platform.openai.com) API key

### 1. Clone the repository

```bash
git clone https://github.com/atsiolis/knowledgebase-ai.git
cd knowledgebase-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

### 4. Set up Supabase

Run the following SQL in your Supabase SQL editor:

```sql
-- Documents table
create table documents (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    created_at timestamp default now()
);

-- Enable pgvector extension
create extension if not exists vector;

-- Chunks table with embeddings
create table chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    content text not null,
    embedding vector(1536),
    metadata jsonb
);

-- Index for fast vector similarity search
create index on chunks using ivfflat (embedding vector_cosine_ops);

-- Similarity search function
create or replace function match_chunks (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    chunks.id,
    chunks.content,
    chunks.metadata,
    1 - (chunks.embedding <=> query_embedding) as similarity
  from chunks
  where 1 - (chunks.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;
```

### 5. Start the backend

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Auto-generated docs at `http://localhost:8000/docs`.

### 6. Open the frontend

Open `frontend/index.html` in your browser, or serve it with:

```bash
npx serve frontend
```

---

## 🛠️ How It Works

1. **Upload** — A document is uploaded via the API. Its name is recorded in the `documents` table and the content is passed to `ingestion.py` to be split into overlapping text chunks using `RecursiveCharacterTextSplitter`.

2. **Embed** — Chunks are sent to OpenAI's `text-embedding-3-small` model in batches of 100 via `ingestion.py`, producing 1536-dimension vectors. This runs in a background thread so uploads don't time out.

3. **Store** — Chunks, embeddings, and metadata are persisted to the `chunks` table via `supabase_client.py`. Deleting a document cascades to remove all its chunks automatically.

4. **Retrieve** — When a question is asked, `retriever.py` embeds it using `AsyncOpenAI` (non-blocking), then calls the `match_chunks` Supabase function via `asyncio.to_thread` to run cosine similarity search without blocking the event loop.

5. **Stream** — The top matching chunks are passed to `llm_chain.py`, which calls GPT-4o with `stream=True` via `AsyncOpenAI`. Tokens are yielded as an async generator and sent to the frontend as Server-Sent Events. The frontend's `ReadableStream` reader renders each token live into the chat bubble as it arrives.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{id}` | Remove a document and its chunks |
| `POST` | `/upload` | Upload a PDF or TXT file |
| `GET` | `/upload/status/{upload_id}` | Poll upload progress |
| `GET` | `/ask?question=...` | Stream an answer via Server-Sent Events |

### SSE Event Format (`/ask`)

The `/ask` endpoint returns a stream of newline-delimited JSON events:

```
data: {"type": "sources", "content": ["report.pdf", "notes.txt"]}

data: {"type": "token", "content": "The"}

data: {"type": "token", "content": " answer"}

data: {"type": "done"}
```

