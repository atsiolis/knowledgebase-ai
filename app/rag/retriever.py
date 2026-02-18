"""
Vector Similarity Retriever

This module handles the "Retrieval" phase of RAG:
1. Convert user's question into a vector embedding
2. Search Supabase for chunks with similar embeddings
3. Return the most relevant chunks

Uses cosine similarity search with pgvector extension in PostgreSQL.
"""

from openai import OpenAI
import os
from app.db.supabase_client import supabase
from app.rag.ingestion import generate_embedding

# Initialize OpenAI client for query embedding
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_similar_chunks(question: str, top_k=5):
    """
    Find document chunks most similar to a given question.
    
    Process:
    1. Generate embedding for the question
    2. Call Supabase RPC function to perform vector similarity search
    3. Return top K most similar chunks with their content and metadata
    
    Args:
        question (str): User's question to search for
        top_k (int): Number of similar chunks to return (default: 5)
        
    Returns:
        list[dict]: List of chunk objects, each containing:
            - id: Chunk ID
            - document_id: Parent document ID
            - content: Text content of the chunk
            - metadata: Source document name and other info
            - similarity: Cosine similarity score (0-1, higher = more similar)
            
    Example:
        chunks = retrieve_similar_chunks("What is RAG?", top_k=3)
        # Returns top 3 chunks most relevant to the question
        
    Note:
        The match_chunks RPC function must exist in Supabase.
        It performs vector similarity search using pgvector's <=> operator.
    """
    
    # Step 1: Generate embedding for the user's question
    # This converts the question into the same 1536-dimensional space as document chunks
    question_emb = generate_embedding(question)
    
    # Step 2: Call Supabase RPC function for similarity search
    # The match_chunks function:
    # - Compares query embedding with all chunk embeddings
    # - Uses cosine similarity (1 - cosine distance)
    # - Filters by similarity threshold
    # - Orders by similarity (highest first)
    # - Returns top K results
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_emb,      # The question's vector
        "match_threshold": 0.2,                # Minimum similarity (0.2 = 20% similar)
        "match_count": top_k                   # How many results to return
    }).execute()
    
    # Step 3: Return the matched chunks
    # Each chunk has content (text) and metadata (source document)
    return result.data
