"""
Shared fixtures for KnowledgeBase AI test suite.

All external dependencies (OpenAI, Supabase) are mocked here so tests
run without any live API keys or network access.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake data constants reused across tests
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536  # Matches text-embedding-3-small dimensions

FAKE_CHUNKS = [
    {
        "id": "chunk-uuid-1",
        "content": "FastAPI is a modern Python web framework.",
        "metadata": {"source": "sample.pdf"},
        "similarity": 0.92,
    },
    {
        "id": "chunk-uuid-2",
        "content": "It supports async/await natively.",
        "metadata": {"source": "sample.pdf"},
        "similarity": 0.85,
    },
]

FAKE_DOCUMENT = {"id": "doc-uuid-1", "name": "sample.pdf"}


# ---------------------------------------------------------------------------
# Supabase mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_supabase():
    """Return a MagicMock that mimics the Supabase client interface."""
    client = MagicMock()

    # documents table
    doc_insert = MagicMock()
    doc_insert.execute.return_value = MagicMock(data=[FAKE_DOCUMENT])
    client.table.return_value.insert.return_value = doc_insert
    client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=[FAKE_DOCUMENT]
    )
    client.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[FAKE_DOCUMENT]
    )

    # chunks table
    chunk_insert = MagicMock()
    chunk_insert.execute.return_value = MagicMock(data=[])
    client.table.return_value.insert.return_value = chunk_insert

    # rpc (match_chunks)
    client.rpc.return_value.execute.return_value = MagicMock(data=FAKE_CHUNKS)

    return client


# ---------------------------------------------------------------------------
# OpenAI sync mock (used by ingestion.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_sync():
    """Mock the synchronous OpenAI client used for batch embeddings."""
    client = MagicMock()
    embedding_obj = MagicMock()
    embedding_obj.embedding = FAKE_EMBEDDING
    client.embeddings.create.return_value = MagicMock(
        data=[embedding_obj, embedding_obj]  # supports batches of 2+
    )
    return client


# ---------------------------------------------------------------------------
# OpenAI async mock (used by retriever.py and llm_chain.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_async():
    """Mock the async OpenAI client used for query embeddings and chat."""
    client = AsyncMock()

    # Embeddings
    embedding_obj = MagicMock()
    embedding_obj.embedding = FAKE_EMBEDDING
    client.embeddings.create.return_value = MagicMock(data=[embedding_obj])

    # Chat completions streaming
    async def fake_stream():
        tokens = ["The", " answer", " is", " 42", "."]
        for token in tokens:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            yield chunk

    client.chat.completions.create.return_value = fake_stream()

    return client


# ---------------------------------------------------------------------------
# FastAPI TestClient (with all external deps patched)
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(mock_supabase, mock_openai_sync, mock_openai_async):
    """
    TestClient with OpenAI and Supabase fully mocked.
    Patches are applied before the app module is imported so module-level
    client initializations pick up the mocks.
    """
    with (
        patch("app.db.supabase_client.supabase", mock_supabase),
        patch("app.rag.ingestion.client", mock_openai_sync),
        patch("app.rag.ingestion.supabase", mock_supabase),
        patch("app.rag.retriever.async_client", mock_openai_async),
        patch("app.rag.retriever.supabase", mock_supabase),
        patch("app.rag.llm_chain.async_client", mock_openai_async),
        patch("app.main.supabase", mock_supabase),
    ):
        from app.main import app
        yield TestClient(app)
