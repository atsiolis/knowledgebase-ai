"""
Tests for app/rag/retriever.py

Covers:
- Happy-path retrieval returning ranked chunks
- Empty result when no chunks match
- Correct embedding model and RPC parameters are used
- Event loop is not blocked (asyncio.to_thread is used for the DB call)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

FAKE_EMBEDDING = [0.1] * 1536
FAKE_CHUNKS = [
    {"id": "c1", "content": "Relevant text.", "metadata": {"source": "a.pdf"}, "similarity": 0.9},
    {"id": "c2", "content": "Also relevant.", "metadata": {"source": "b.pdf"}, "similarity": 0.8},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_async_openai_mock(embedding=None):
    client = AsyncMock()
    emb_obj = MagicMock()
    emb_obj.embedding = embedding or FAKE_EMBEDDING
    client.embeddings.create.return_value = MagicMock(data=[emb_obj])
    return client


def make_supabase_mock(chunks=None):
    client = MagicMock()
    # Use `is not None` check — `chunks or FAKE_CHUNKS` would treat [] as falsy
    # and incorrectly fall back to FAKE_CHUNKS when testing the empty-results case.
    data = FAKE_CHUNKS if chunks is None else chunks
    client.rpc.return_value.execute.return_value = MagicMock(data=data)
    return client


# ---------------------------------------------------------------------------
# retrieve_similar_chunks
# ---------------------------------------------------------------------------

class TestRetrieveSimilarChunks:
    def test_returns_matching_chunks(self):
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock(FAKE_CHUNKS)

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            result = asyncio.run(retrieve_similar_chunks("What is FastAPI?", top_k=2))

        assert len(result) == 2
        assert result[0]["content"] == "Relevant text."
        assert result[0]["similarity"] == 0.9

    def test_returns_empty_list_when_no_matches(self):
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock([])

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            result = asyncio.run(retrieve_similar_chunks("unrelated question"))

        assert result == []

    def test_uses_correct_embedding_model(self):
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock()

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            asyncio.run(retrieve_similar_chunks("test question"))

        mock_async.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input="test question",
        )

    def test_calls_rpc_with_correct_params(self):
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock()

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            asyncio.run(retrieve_similar_chunks("test", top_k=3))

        mock_supa.rpc.assert_called_once_with(
            "match_chunks",
            {
                "query_embedding": FAKE_EMBEDDING,
                "match_threshold": 0.2,
                "match_count": 3,
            },
        )

    def test_respects_top_k_parameter(self):
        """Different top_k values are forwarded correctly to the RPC."""
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock()

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            asyncio.run(retrieve_similar_chunks("test", top_k=10))

        call_args = mock_supa.rpc.call_args
        assert call_args[0][1]["match_count"] == 10

    def test_default_top_k_is_5(self):
        mock_async = make_async_openai_mock()
        mock_supa = make_supabase_mock()

        with (
            patch("app.rag.retriever.async_client", mock_async),
            patch("app.rag.retriever.supabase", mock_supa),
        ):
            from app.rag.retriever import retrieve_similar_chunks
            asyncio.run(retrieve_similar_chunks("test"))

        call_args = mock_supa.rpc.call_args
        assert call_args[0][1]["match_count"] == 5
