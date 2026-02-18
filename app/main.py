"""
FastAPI backend for KnowledgeBase AI - Document Q&A System

This module handles:
- File uploads and document processing
- Vector similarity search for Q&A
- Real-time progress tracking for uploads
- Document management (list, delete)

Tech Stack: FastAPI, Supabase (pgvector), OpenAI
"""

import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.db.supabase_client import supabase
from app.rag.ingestion import extract_text, chunk_text, generate_embeddings_batch
from app.rag.retriever import retrieve_similar_chunks
from app.rag.llm_chain import generate_answer
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime

# Initialize FastAPI application
app = FastAPI()

# In-memory dictionary to track upload progress
# Key: upload_id (UUID), Value: progress data (status, progress %, message, etc.)
upload_progress = {}

# Configure CORS to allow frontend to communicate with backend
# In production, replace "*" with specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],   # Allows all HTTP methods
    allow_headers=["*"],   # Allows all headers
)


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
def health():
    """
    Simple health check endpoint to verify API is running.
    
    Returns:
        dict: Status message
    """
    return {"status": "ok"}


# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/documents")
def get_documents():
    """
    Retrieve all documents from the database.
    
    Returns:
        list: Array of document objects with id, name, created_at
    """
    response = supabase.table("documents").select("*").execute()
    return response.data 

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks (cascade delete).
    
    Args:
        document_id (str): UUID of the document to delete
        
    Returns:
        dict: Status and message indicating success or failure
    """
    try:
        # Delete the document from Supabase
        # Chunks are automatically deleted due to ON DELETE CASCADE foreign key
        response = supabase.table("documents").delete().eq("id", document_id).execute()
        
        if response.data:
            return {
                "status": "success", 
                "message": f"Document deleted successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Document not found"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================================
# UPLOAD & PROCESSING ENDPOINTS
# ============================================================================

@app.get("/upload/status/{upload_id}")
def get_upload_status(upload_id: str):
    """
    Get real-time progress of a document upload/processing.
    
    Frontend polls this endpoint every 500ms to display progress bar.
    
    Args:
        upload_id (str): Unique identifier for the upload
        
    Returns:
        dict: Progress data including status, progress %, message, chunk counts
    """
    if upload_id not in upload_progress:
        return {"status": "not_found"}
    return upload_progress[upload_id]

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Handle file upload and trigger background processing.
    
    Process:
    1. Save uploaded file temporarily
    2. Generate unique upload ID
    3. Initialize progress tracking
    4. Queue background processing task
    5. Return immediately (non-blocking)
    
    Args:
        background_tasks: FastAPI background task manager
        file: Uploaded file (PDF or TXT)
        
    Returns:
        dict: Upload ID and status for progress tracking
    """
    file_path = f"../temp_{file.filename}"
    upload_id = str(uuid.uuid4())  # Generate unique ID for this upload
    
    # Initialize progress tracking with starting values
    upload_progress[upload_id] = {
        "status": "uploading",
        "filename": file.filename,
        "progress": 0,
        "message": "Uploading file...",
        "total_chunks": 0,
        "processed_chunks": 0
    }
    
    try:
        # Save uploaded file to temporary location
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Update progress: file saved
        upload_progress[upload_id]["status"] = "processing"
        upload_progress[upload_id]["progress"] = 10
        upload_progress[upload_id]["message"] = "File uploaded, starting processing..."
        
        # Add processing task to background queue (non-blocking)
        # This allows the endpoint to return immediately
        background_tasks.add_task(process_file_background, file.filename, file_path, upload_id)
        
        return {
            "status": "processing", 
            "upload_id": upload_id,
            "message": "File is being processed in the background"
        }
    
    except Exception as e:
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["message"] = str(e)
        return {"status": "error", "detail": str(e)}


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_file_background(filename: str, file_path: str, upload_id: str):
    """
    Process uploaded file in background to avoid timeouts.
    
    Steps:
    1. Extract text from PDF/TXT (20%)
    2. Split into chunks (30%)
    3. Generate embeddings (40-70%)
    4. Save to database (70-100%)
    
    Updates progress at each step for real-time UI updates.
    
    Args:
        filename: Original name of the uploaded file
        file_path: Temporary path where file is stored
        upload_id: Unique identifier for progress tracking
    """
    try:
        # Step 1: Extract text from document
        upload_progress[upload_id]["progress"] = 20
        upload_progress[upload_id]["message"] = "Extracting text from document..."
        text = extract_text(file_path)
        
        # Step 2: Split text into chunks (~800 chars each)
        upload_progress[upload_id]["progress"] = 30
        upload_progress[upload_id]["message"] = "Splitting into chunks..."
        chunks = chunk_text(text)
        upload_progress[upload_id]["total_chunks"] = len(chunks)
        
        # Step 3 & 4: Generate embeddings and save to database
        upload_progress[upload_id]["progress"] = 40
        upload_progress[upload_id]["message"] = f"Generating embeddings for {len(chunks)} chunks..."
        
        save_chunks_with_progress(filename, chunks, upload_id)
        
        # Mark as complete
        upload_progress[upload_id]["status"] = "complete"
        upload_progress[upload_id]["progress"] = 100
        upload_progress[upload_id]["message"] = f"✅ Successfully processed {len(chunks)} chunks!"
        
        print(f"✅ Successfully processed {filename}: {len(chunks)} chunks")
        
    except Exception as e:
        # Handle any errors during processing
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["progress"] = 0
        upload_progress[upload_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Error processing {filename}: {str(e)}")
        
    finally:
        # Always clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

def save_chunks_with_progress(document_name: str, chunks: list, upload_id: str):
    """
    Save document chunks to database with real-time progress updates.
    
    Process:
    1. Create document entry in database
    2. Generate embeddings in batches (100 chunks at a time)
    3. Insert chunks to database in batches (50 at a time)
    4. Update progress after each batch
    
    Args:
        document_name: Name of the document
        chunks: List of text chunks
        upload_id: Upload ID for progress tracking
    """
    
    # Create document entry and get its ID
    doc_resp = supabase.table("documents").insert({"name": document_name}).execute()
    document_id = doc_resp.data[0]["id"]
    
    # Batch sizes optimized to avoid timeouts
    EMBEDDING_BATCH_SIZE = 100  # OpenAI can handle up to 2048, but 100 is safer
    SUPABASE_BATCH_SIZE = 50    # Smaller batches avoid Supabase timeout with large vectors
    
    all_rows = []
    total_chunks = len(chunks)
    
    # ========================================================================
    # PHASE 1: Generate embeddings in batches (Progress: 40-70%)
    # ========================================================================
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        # Generate embeddings for entire batch in one API call (much faster!)
        embeddings = generate_embeddings_batch(batch)
        
        # Prepare rows for database insertion
        for chunk, embedding in zip(batch, embeddings):
            all_rows.append({
                "document_id": document_id,
                "content": chunk,
                "embedding": embedding,
                "metadata": {"source": document_name}
            })
        
        # Update progress (40-70% range for embeddings)
        progress = 40 + int((len(all_rows) / total_chunks) * 30)
        upload_progress[upload_id]["progress"] = progress
        upload_progress[upload_id]["message"] = f"Generated {len(all_rows)}/{total_chunks} embeddings..."
    
    # ========================================================================
    # PHASE 2: Insert to database in batches (Progress: 70-100%)
    # ========================================================================
    upload_progress[upload_id]["message"] = "Saving to database..."
    
    for i in range(0, len(all_rows), SUPABASE_BATCH_SIZE):
        batch_rows = all_rows[i:i + SUPABASE_BATCH_SIZE]
        
        # Retry logic for transient network failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Insert batch to Supabase
                supabase.table("chunks").insert(batch_rows).execute()
                
                # Update progress (70-100% range for database inserts)
                chunks_saved = min(i + SUPABASE_BATCH_SIZE, len(all_rows))
                progress = 70 + int((chunks_saved / len(all_rows)) * 30)
                upload_progress[upload_id]["progress"] = progress
                upload_progress[upload_id]["processed_chunks"] = chunks_saved
                upload_progress[upload_id]["message"] = f"Saved {chunks_saved}/{len(all_rows)} chunks to database..."
                break  # Success - exit retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # Give up after max retries
                import time
                time.sleep(1)  # Wait before retry


# ============================================================================
# Q&A ENDPOINT
# ============================================================================

@app.get("/ask")
async def ask_question(question: str):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    Process:
    1. Generate embedding for the question
    2. Search for similar chunks in vector database
    3. Send relevant chunks + question to LLM
    4. Return answer with source citations
    
    Args:
        question (str): User's question
        
    Returns:
        dict: Answer and list of source documents
    """
    # Step 1: Find relevant text chunks using vector similarity search
    relevant_chunks = retrieve_similar_chunks(question, top_k=3)
    
    if not relevant_chunks:
        return {"answer": "No relevant documents found. Please upload some files first."}

    # Step 2: Generate answer using LLM with retrieved context
    answer = generate_answer(question, relevant_chunks)
    
    return {
        "answer": answer,
        "sources": [c.get('metadata', {}).get('source', 'Unknown') for c in relevant_chunks]
    }
