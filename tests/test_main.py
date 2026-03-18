"""
Integration tests for app/main.py (FastAPI endpoints).

Uses FastAPI's TestClient so the full request/response cycle is exercised,
but all external I/O (OpenAI, Supabase, filesystem) is mocked.

Endpoints covered:
- GET  /health
- GET  /documents
- DELETE /documents/{id}
- POST /upload
- GET  /upload/status/{upload_id}
- GET  /ask
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1536
FAKE_DOCUMENT = {"id": "doc-uuid-1", "name": "sample.pdf"}
FAKE_CHUNKS = [
    {"id": "c1", "content": "Relevant.", "metadata": {"source": "sample.pdf"}, "similarity": 0.9},
]


def _make_supabase():
    sb = MagicMock()
    sb.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=[FAKE_DOCUMENT]
    )
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[FAKE_DOCUMENT]
    )
    sb.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[FAKE_DOCUMENT]
    )
    sb.rpc.return_value.execute.return_value = MagicMock(data=FAKE_CHUNKS)
    return sb


def _make_async_openai():
    client = AsyncMock()
    emb = MagicMock()
    emb.embedding = FAKE_EMBEDDING
    client.embeddings.create.return_value = MagicMock(data=[emb])

    async def fake_stream():
        for token in ["The", " answer", "."]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            yield chunk

    client.chat.completions.create.return_value = fake_stream()
    return client


@pytest.fixture
def client():
    mock_sb = _make_supabase()
    mock_async_oai = _make_async_openai()
    mock_sync_oai = MagicMock()
    emb = MagicMock()
    emb.embedding = FAKE_EMBEDDING
    mock_sync_oai.embeddings.create.return_value = MagicMock(data=[emb])

    with (
        patch("app.db.supabase_client.supabase", mock_sb),
        patch("app.rag.ingestion.client", mock_sync_oai),
        patch("app.rag.ingestion.supabase", mock_sb),
        patch("app.rag.retriever.async_client", mock_async_oai),
        patch("app.rag.retriever.supabase", mock_sb),
        patch("app.rag.llm_chain.async_client", mock_async_oai),
        patch("app.main.supabase", mock_sb),
    ):
        # Import app fresh inside the patch context
        import importlib

        import app.main as main_module
        importlib.reload(main_module)
        yield TestClient(main_module.app), mock_sb, mock_async_oai


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_ok(self, client):
        tc, *_ = client
        response = tc.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

class TestGetDocuments:
    def test_returns_document_list(self, client):
        tc, *_ = client
        response = tc.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert data[0]["name"] == "sample.pdf"

    def test_returns_empty_list_when_no_documents(self, client):
        tc, mock_sb, _ = client
        mock_sb.table.return_value.select.return_value.execute.return_value = MagicMock(data=[])

        response = tc.get("/documents")
        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# DELETE /documents/{id}
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_successful_deletion(self, client):
        tc, *_ = client
        response = tc.delete("/documents/doc-uuid-1")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_not_found_returns_error(self, client):
        tc, mock_sb, _ = client
        mock_sb.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[]
        )
        response = tc.delete("/documents/nonexistent-id")
        assert response.status_code == 200
        assert response.json()["status"] == "error"


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    def test_returns_upload_id(self, client, tmp_path):
        tc, *_ = client
        txt = tmp_path / "test.txt"
        txt.write_text("Some content here.", encoding="utf-8")

        with (
            patch("app.main.process_file_background"),  # Skip background processing
            open(txt, "rb") as f,
        ):
            response = tc.post("/upload", files={"file": ("test.txt", f, "text/plain")})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "processing"
        assert "upload_id" in body

    def test_upload_id_is_uuid_format(self, client, tmp_path):
        tc, *_ = client
        txt = tmp_path / "file.txt"
        txt.write_text("content", encoding="utf-8")

        with (
            patch("app.main.process_file_background"),
            open(txt, "rb") as f,
        ):
            response = tc.post("/upload", files={"file": ("file.txt", f, "text/plain")})

        upload_id = response.json()["upload_id"]
        import uuid
        # Should not raise
        uuid.UUID(upload_id)


# ---------------------------------------------------------------------------
# GET /upload/status/{upload_id}
# ---------------------------------------------------------------------------

class TestUploadStatus:
    def test_not_found_for_unknown_id(self, client):
        tc, *_ = client
        response = tc.get("/upload/status/nonexistent-id")
        assert response.status_code == 200
        assert response.json()["status"] == "not_found"

    def test_returns_progress_for_known_id(self, client, tmp_path):
        tc, *_ = client
        txt = tmp_path / "prog.txt"
        txt.write_text("content", encoding="utf-8")

        with (
            patch("app.main.process_file_background"),
            open(txt, "rb") as f,
        ):
            upload_resp = tc.post("/upload", files={"file": ("prog.txt", f, "text/plain")})

        upload_id = upload_resp.json()["upload_id"]
        status_resp = tc.get(f"/upload/status/{upload_id}")

        assert status_resp.status_code == 200
        body = status_resp.json()
        assert "status" in body
        assert "progress" in body


# ---------------------------------------------------------------------------
# GET /ask  (SSE streaming endpoint)
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    def _parse_sse(self, text):
        """Parse raw SSE text into a list of event dicts."""
        events = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def test_streams_sources_and_tokens(self, client):
        tc, *_ = client
        response = tc.get("/ask?question=What+is+FastAPI")

        assert response.status_code == 200
        events = self._parse_sse(response.text)

        types = [e["type"] for e in events]
        assert "sources" in types
        assert "token" in types
        assert "done" in types

    def test_sources_event_contains_source_name(self, client):
        tc, *_ = client
        response = tc.get("/ask?question=What+is+FastAPI")
        events = self._parse_sse(response.text)

        sources_events = [e for e in events if e["type"] == "sources"]
        assert len(sources_events) == 1
        assert "sample.pdf" in sources_events[0]["content"]

    def test_done_event_is_last(self, client):
        tc, *_ = client
        response = tc.get("/ask?question=test")
        events = self._parse_sse(response.text)

        assert events[-1]["type"] == "done"

    def test_no_documents_returns_error_event(self, client):
        tc, mock_sb, _ = client
        # Make retrieval return nothing
        mock_sb.rpc.return_value.execute.return_value = MagicMock(data=[])

        response = tc.get("/ask?question=anything")
        events = self._parse_sse(response.text)

        assert any(e["type"] == "error" for e in events)

    def test_question_must_be_provided(self, client):
        tc, *_ = client
        # FastAPI returns 422 for missing required query params
        response = tc.get("/ask")
        assert response.status_code == 422

    def test_content_type_is_event_stream(self, client):
        tc, *_ = client
        response = tc.get("/ask?question=test")
        assert "text/event-stream" in response.headers.get("content-type", "")
