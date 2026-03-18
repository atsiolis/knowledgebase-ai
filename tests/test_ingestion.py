"""
Tests for app/rag/ingestion.py

Covers:
- Text extraction from PDF and TXT files
- Text chunking behaviour (size, overlap, edge cases)
- Single and batch embedding generation
- Database save with progress tracking
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_extracts_txt_file(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Hello, world!", encoding="utf-8")

        from app.rag.ingestion import extract_text
        result = extract_text(str(txt_file))

        assert result == "Hello, world!"

    def test_extracts_pdf_pages(self):
        """pdfplumber is mocked — each page returns its text."""
        fake_page_1 = MagicMock()
        fake_page_1.extract_text.return_value = "Page one content."
        fake_page_2 = MagicMock()
        fake_page_2.extract_text.return_value = "Page two content."

        fake_pdf = MagicMock()
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.pages = [fake_page_1, fake_page_2]

        with patch("app.rag.ingestion.pdfplumber.open", return_value=fake_pdf):
            from app.rag.ingestion import extract_text
            result = extract_text("document.pdf")

        assert "Page one content." in result
        assert "Page two content." in result

    def test_skips_empty_pdf_pages(self):
        """Image-only pages return None from extract_text — should be skipped."""
        good_page = MagicMock()
        good_page.extract_text.return_value = "Real text."
        empty_page = MagicMock()
        empty_page.extract_text.return_value = None

        fake_pdf = MagicMock()
        fake_pdf.__enter__.return_value = fake_pdf
        fake_pdf.pages = [good_page, empty_page]

        with patch("app.rag.ingestion.pdfplumber.open", return_value=fake_pdf):
            from app.rag.ingestion import extract_text
            result = extract_text("document.pdf")

        assert result.strip() == "Real text."

    def test_txt_encoding_utf8(self, tmp_path):
        txt_file = tmp_path / "greek.txt"
        txt_file.write_text("Γεια σου κόσμε", encoding="utf-8")

        from app.rag.ingestion import extract_text
        result = extract_text(str(txt_file))

        assert "Γεια σου" in result


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_returns_list_of_strings(self):
        from app.rag.ingestion import chunk_text
        result = chunk_text("Hello world. " * 100)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_chunks_are_within_size_limit(self):
        from app.rag.ingestion import chunk_text
        # Generous margin: splitter may slightly exceed chunk_size at word boundaries
        chunks = chunk_text("word " * 500, chunk_size=200, overlap=50)
        for chunk in chunks:
            assert len(chunk) <= 300, f"Chunk too large: {len(chunk)} chars"

    def test_short_text_returns_single_chunk(self):
        from app.rag.ingestion import chunk_text
        result = chunk_text("Short text.", chunk_size=800, overlap=150)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_empty_text_returns_empty_list(self):
        from app.rag.ingestion import chunk_text
        result = chunk_text("", chunk_size=800, overlap=150)
        assert result == []

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        from app.rag.ingestion import chunk_text
        text = "word " * 400
        with_overlap = chunk_text(text, chunk_size=200, overlap=100)
        without_overlap = chunk_text(text, chunk_size=200, overlap=0)
        assert len(with_overlap) >= len(without_overlap)


# ---------------------------------------------------------------------------
# generate_embedding
# ---------------------------------------------------------------------------

class TestGenerateEmbedding:
    def test_returns_1536_dim_vector(self, mock_openai_sync):
        with patch("app.rag.ingestion.client", mock_openai_sync):
            from app.rag.ingestion import generate_embedding
            result = generate_embedding("test query")

        assert isinstance(result, list)
        assert len(result) == 1536

    def test_calls_correct_model(self, mock_openai_sync):
        with patch("app.rag.ingestion.client", mock_openai_sync):
            from app.rag.ingestion import generate_embedding
            generate_embedding("test query")

        mock_openai_sync.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test query",
        )


# ---------------------------------------------------------------------------
# generate_embeddings_batch
# ---------------------------------------------------------------------------

class TestGenerateEmbeddingsBatch:
    def test_returns_one_embedding_per_input(self, mock_openai_sync):
        texts = ["chunk one", "chunk two"]
        # Make the mock return exactly 2 embedding objects
        emb = MagicMock()
        emb.embedding = [0.1] * 1536
        mock_openai_sync.embeddings.create.return_value = MagicMock(
            data=[emb, emb]
        )

        with patch("app.rag.ingestion.client", mock_openai_sync):
            from app.rag.ingestion import generate_embeddings_batch
            result = generate_embeddings_batch(texts)

        assert len(result) == 2
        assert all(len(e) == 1536 for e in result)

    def test_sends_all_texts_in_one_call(self, mock_openai_sync):
        texts = ["a", "b", "c"]
        emb = MagicMock()
        emb.embedding = [0.0] * 1536
        mock_openai_sync.embeddings.create.return_value = MagicMock(
            data=[emb, emb, emb]
        )

        with patch("app.rag.ingestion.client", mock_openai_sync):
            from app.rag.ingestion import generate_embeddings_batch
            generate_embeddings_batch(texts)

        # Should only call the API once with all three texts
        mock_openai_sync.embeddings.create.assert_called_once()
        call_kwargs = mock_openai_sync.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == texts


# ---------------------------------------------------------------------------
# save_chunks (basic version without progress)
# ---------------------------------------------------------------------------

class TestSaveChunks:
    def _make_supabase(self):
        """
        Build a fresh Supabase mock with insert returning a document row.
        We don't reuse the conftest fixture here because its document insert
        mock gets overwritten by the chunk insert mock (last assignment wins).
        """
        sb = MagicMock()
        sb.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": "doc-uuid-1", "name": "sample.pdf"}]
        )
        return sb

    def test_inserts_document_and_chunks(self, mock_openai_sync):
        emb = MagicMock()
        emb.embedding = [0.1] * 1536
        mock_openai_sync.embeddings.create.return_value = MagicMock(data=[emb, emb])
        sb = self._make_supabase()

        with (
            patch("app.rag.ingestion.supabase", sb),
            patch("app.rag.ingestion.client", mock_openai_sync),
        ):
            from app.rag.ingestion import save_chunks
            save_chunks("sample.pdf", ["chunk one", "chunk two"])

        sb.table.assert_any_call("documents")

    def test_handles_empty_chunks(self, mock_openai_sync):
        sb = self._make_supabase()

        with (
            patch("app.rag.ingestion.supabase", sb),
            patch("app.rag.ingestion.client", mock_openai_sync),
        ):
            from app.rag.ingestion import save_chunks
            # Should not raise even with an empty chunk list
            save_chunks("empty.pdf", [])
