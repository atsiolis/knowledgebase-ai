"""
Tests for app/rag/llm_chain.py

Covers:
- Tokens are yielded in the correct order
- Multiple chunks are combined into a single context string
- Empty chunks list is handled gracefully
- GPT-4o is called with temperature=0 and stream=True
- System prompt contains the context and question
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stream(*tokens):
    """Build an async generator that yields chat-completion-style chunks."""
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


def make_async_client_mock(*tokens):
    client = AsyncMock()
    client.chat.completions.create.return_value = make_stream(*tokens)
    return client


SAMPLE_CHUNKS = [
    {"content": "FastAPI supports async.", "metadata": {"source": "doc.pdf"}},
    {"content": "It uses Starlette under the hood.", "metadata": {"source": "doc.pdf"}},
]


# ---------------------------------------------------------------------------
# generate_answer_stream
# ---------------------------------------------------------------------------

class TestGenerateAnswerStream:
    def _collect(self, async_gen):
        """Collect all tokens from an async generator synchronously."""
        async def _gather():
            tokens = []
            async for token in async_gen:
                tokens.append(token)
            return tokens
        return asyncio.run(_gather())

    def test_yields_tokens_in_order(self):
        mock_client = make_async_client_mock("Hello", " world", "!")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            tokens = self._collect(generate_answer_stream("test?", SAMPLE_CHUNKS))

        assert tokens == ["Hello", " world", "!"]

    def test_joined_tokens_form_complete_answer(self):
        mock_client = make_async_client_mock("The", " answer", " is", " 42.")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            tokens = self._collect(generate_answer_stream("What is the answer?", SAMPLE_CHUNKS))

        assert "".join(tokens) == "The answer is 42."

    def test_none_tokens_are_filtered(self):
        """OpenAI sometimes yields chunks with delta.content = None (e.g. the last chunk)."""
        async def stream_with_none():
            for content in ["Hello", None, " world"]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = content
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = stream_with_none()

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            tokens = self._collect(generate_answer_stream("q?", SAMPLE_CHUNKS))

        assert None not in tokens
        assert "".join(tokens) == "Hello world"

    def test_uses_gpt4o_with_streaming(self):
        mock_client = make_async_client_mock("ok")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            self._collect(generate_answer_stream("test?", SAMPLE_CHUNKS))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["stream"] is True
        assert call_kwargs["temperature"] == 0

    def test_context_contains_all_chunk_content(self):
        mock_client = make_async_client_mock("ok")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            self._collect(generate_answer_stream("test?", SAMPLE_CHUNKS))

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "FastAPI supports async." in prompt
        assert "It uses Starlette under the hood." in prompt

    def test_question_is_in_prompt(self):
        mock_client = make_async_client_mock("ok")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            self._collect(generate_answer_stream("What is pgvector?", SAMPLE_CHUNKS))

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "What is pgvector?" in prompt

    def test_empty_chunks_does_not_crash(self):
        mock_client = make_async_client_mock("I don't know.")

        with patch("app.rag.llm_chain.async_client", mock_client):
            from app.rag.llm_chain import generate_answer_stream
            tokens = self._collect(generate_answer_stream("anything?", []))

        assert "".join(tokens) == "I don't know."
