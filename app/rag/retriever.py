"""
Vector Similarity Retriever

This module handles the "Retrieval" phase of RAG:
1. Asynchronously embed the user's question via AsyncOpenAI
2. Run the blocking Supabase RPC call in a thread pool (asyncio.to_thread)
   so the FastAPI event loop is never blocked
3. Return the most relevant chunks

Uses cosine similarity search with the pgvector extension in PostgreSQL.
"""

import asyncio
import os
from openai import AsyncOpenAI
from app.db.supabase_client import supabase

# Async OpenAI client — used here only for query embeddings
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def retrieve_similar_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Asynchronously find document chunks most similar to a given question.

    Both the embedding call and the Supabase query are non-blocking:
    - OpenAI embedding uses AsyncOpenAI (native async)
    - Supabase RPC (sync library) is offloaded to a thread via asyncio.to_thread,
      freeing the event loop to handle other requests while the DB query runs

    Args:
        question (str): User's question to search for
        top_k (int): Number of similar chunks to return (default: 5)

    Returns:
        list[dict]: List of chunk objects, each containing:
            - id: Chunk UUID
            - content: Text content of the chunk
            - metadata: Source document name and other info
            - similarity: Cosine similarity score (0–1, higher = more similar)
    """

    # Step 1: Embed the question (async — does not block the event loop)
    embedding_response = await async_client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    question_emb = embedding_response.data[0].embedding

    # Step 2: Run the blocking Supabase RPC in a thread pool
    # asyncio.to_thread wraps the synchronous call so FastAPI can keep serving
    # other requests while this DB round-trip completes
    result = await asyncio.to_thread(
        lambda: supabase.rpc(
            "match_chunks",
            {
                "query_embedding": question_emb,
                "match_threshold": 0.2,   # Minimum cosine similarity to qualify
                "match_count": top_k,     # Max number of results to return
            },
        ).execute()
    )

    return result.data