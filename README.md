# 🧠 KnowledgeBase AI

An AI-powered document Q&A system with a real-time RAG (Retrieval-Augmented Generation) pipeline. Upload PDFs or text files and get instant, accurate answers grounded in your documents — complete with source citations.

---

## ✨ Features

- **Document Ingestion** — Upload PDFs or plain text files and have them automatically chunked, embedded, and stored
- **Semantic Search** — Uses OpenAI embeddings and Supabase's pgvector store to find the most relevant passages for any query
- **Cited Answers** — Every answer references the exact source chunks it was derived from, so you can verify and trace back to the original content
- **Real-time RAG Pipeline** — From upload to queryable knowledge in seconds
- **Clean Frontend** — A simple, intuitive HTML/CSS/JS interface for uploading documents and asking questions

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | Supabase (pgvector) |
| LLM | OpenAI GPT |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
knowledgebase-ai/
├── app/
│   ├── db/
│   │   └── supabase_client.py    # Supabase connection & client setup
│   ├── rag/
│   │   ├── ingestion.py          # Document parsing & chunking
│   │   ├── llm_chain.py          # LLM prompt chain & answer generation
│   │   └── retriever.py          # Vector similarity search
│   └── main.py                   # FastAPI entry point & route definitions
├── frontend/
│   ├── index.html                # Main UI
│   ├── scripts.js                # Frontend logic
│   └── style.css                 # Styling
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

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

### 4. Set up Supabase

In your Supabase project, run the following SQL to create the required tables, enable vector support, and register the similarity search function:

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

The API will be available at `http://localhost:8000`. You can explore the auto-generated docs at `http://localhost:8000/docs`.

### 6. Open the frontend

Open `frontend/index.html` directly in your browser, or serve it with any static file server:

```bash
npx serve frontend
```

---

## 🛠️ How It Works

1. **Upload** — A document is uploaded via the API. Its name is recorded in the `documents` table and the content is passed to `ingestion.py` to be split into overlapping text chunks.
2. **Embed** — Each chunk is sent to OpenAI's embedding model via `llm_chain.py` to produce a 1536-dimension vector.
3. **Store** — Chunks, their embeddings, and metadata are persisted to the `chunks` table in Supabase through `supabase_client.py`. Deleting a document cascades to remove all its associated chunks automatically.
4. **Query** — A user's question is embedded using the same model, then `retriever.py` calls the `match_chunks` Supabase function to perform cosine similarity search, returning the most relevant chunks above a configurable threshold.
5. **Generate** — The retrieved chunks are injected as context into an OpenAI chat prompt via `llm_chain.py`. The model returns a grounded answer with source citations.
