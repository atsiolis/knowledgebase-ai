"""
LLM Chain for Answer Generation

This module handles the final step of RAG:
- Takes retrieved document chunks
- Constructs a prompt with context
- Streams GPT-4o tokens back as an async generator

Uses AsyncOpenAI so the FastAPI event loop is never blocked.
"""

from openai import AsyncOpenAI
import os
from typing import AsyncGenerator

# Async OpenAI client — non-blocking, compatible with FastAPI's event loop
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_answer_stream(question: str, chunks: list) -> AsyncGenerator[str, None]:
    """
    Stream an answer token-by-token using GPT-4o.

    Implements the "Generation" step of RAG. The LLM is given relevant
    document chunks as context and streams its response back, allowing
    the frontend to render tokens as they arrive rather than waiting
    for the full response.

    Args:
        question (str): The user's question
        chunks (list): Relevant document chunks from vector search.
                       Each chunk is a dict with 'content' and 'metadata'.

    Yields:
        str: Individual text tokens as they are produced by the model.

    Example:
        async for token in generate_answer_stream("What is RAG?", chunks):
            print(token, end="", flush=True)
    """

    # Combine all chunk contents into a single context string
    context = "\n\n".join([c["content"] for c in chunks])

    prompt = f"""You are an assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer concisely, and include citations if possible.
Use markdown for formatting."""

    # stream=True tells OpenAI to return an async iterator of chunks
    stream = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )

    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            yield token