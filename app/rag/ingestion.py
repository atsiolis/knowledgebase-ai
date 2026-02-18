"""
Document Ingestion Pipeline

This module handles the "Retrieval" preparation phase of RAG:
1. Extract text from PDF or TXT files
2. Split text into manageable chunks
3. Generate vector embeddings for each chunk
4. Save chunks and embeddings to Supabase

Key optimizations:
- Batch embedding generation (100 chunks at once)
- Batch database inserts (50 rows at once)
- Retry logic for network failures
"""

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from app.db.supabase_client import supabase

# Initialize OpenAI client for embedding generation
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text(file_path: str) -> str:
    """
    Extract text content from a PDF or TXT file.
    
    For PDFs: Uses pdfplumber to extract text from each page
    For TXT: Simply reads the file content
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Extracted text content
        
    Note:
        Empty pages in PDFs are skipped (e.g., image-only pages)
    """
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                # Only add non-empty pages (skip image-only pages)
                if extracted:
                    text += extracted + "\n"
        return text
    else:
        # Handle TXT files
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def chunk_text(text: str, chunk_size=800, overlap=150):
    """
    Split text into smaller chunks for embedding generation.
    
    Uses RecursiveCharacterTextSplitter which:
    - Tries to split on paragraphs first
    - Falls back to sentences if paragraphs are too long
    - Maintains overlap between chunks for context continuity
    
    Args:
        text (str): Full document text
        chunk_size (int): Target size of each chunk in characters (default: 800)
        overlap (int): Number of characters to overlap between chunks (default: 150)
        
    Returns:
        list[str]: List of text chunks
        
    Why these sizes?
    - 800 chars ≈ 150-200 tokens (well within embedding model limits)
    - 150 char overlap ensures context isn't lost between chunks
    - Smaller chunks = more precise retrieval
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embedding(text: str):
    """
    Generate a vector embedding for a single text string.
    
    Used by the retriever for query embeddings.
    For document chunks, use generate_embeddings_batch() instead (faster).
    
    Args:
        text (str): Text to embed
        
    Returns:
        list[float]: 1536-dimensional embedding vector
        
    Model: text-embedding-3-small
    - Fast and cost-effective
    - 1536 dimensions
    - Good for semantic search
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    This is MUCH faster than calling generate_embedding() in a loop:
    - 100 chunks: ~2 seconds vs ~30 seconds
    - Reduces API calls by 100x
    - Lower cost (same price per token, but fewer API overhead)
    
    Args:
        texts (list[str]): List of text strings to embed
        
    Returns:
        list[list[float]]: List of embedding vectors, same order as input
        
    Note:
        OpenAI supports up to 2048 texts per request, but we batch
        at 100 for safety and better progress tracking.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts  # Send all texts at once!
    )
    # Extract embeddings in the same order as input
    return [item.embedding for item in response.data]


# ============================================================================
# DATABASE STORAGE
# ============================================================================

def save_chunks(document_name: str, chunks: list):
    """
    Save document chunks and their embeddings to Supabase.
    
    This is the basic version used when not tracking progress.
    For progress tracking, main.py uses save_chunks_with_progress().
    
    Process:
    1. Create document entry in 'documents' table
    2. Generate embeddings for all chunks (batched)
    3. Insert chunks with embeddings to 'chunks' table (batched)
    
    Args:
        document_name (str): Name of the document
        chunks (list[str]): List of text chunks
        
    Database Schema:
        documents: id (uuid), name (text), created_at (timestamp)
        chunks: id (bigint), document_id (uuid FK), content (text),
                embedding (vector 1536), metadata (jsonb)
    """
    
    # Step 1: Create document entry and get its ID
    doc_resp = supabase.table("documents").insert({"name": document_name}).execute()
    document_id = doc_resp.data[0]["id"]
    
    # Batch sizes optimized to avoid timeouts
    EMBEDDING_BATCH_SIZE = 100  # OpenAI batch size
    SUPABASE_BATCH_SIZE = 50    # Supabase insert batch size (smaller due to vector size)
    
    all_rows = []
    
    # Step 2: Generate embeddings in batches
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        embeddings = generate_embeddings_batch(batch)
        
        # Prepare rows for database insertion
        for chunk, embedding in zip(batch, embeddings):
            all_rows.append({
                "document_id": document_id,
                "content": chunk,
                "embedding": embedding,
                "metadata": {"source": document_name}  # Track source document
            })
    
    # Step 3: Insert to database in batches with retry logic
    for i in range(0, len(all_rows), SUPABASE_BATCH_SIZE):
        batch_rows = all_rows[i:i + SUPABASE_BATCH_SIZE]
        
        # Retry up to 3 times for transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                supabase.table("chunks").insert(batch_rows).execute()
                break  # Success - move to next batch
            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # Give up after max retries
                import time
                time.sleep(1)  # Wait before retry
