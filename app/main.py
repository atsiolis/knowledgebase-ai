"""
FastAPI backend for KnowledgeBase AI - Document Q&A System

This module handles:
- File uploads and document processing
- Streaming Q&A via Server-Sent Events (SSE)
- Real-time progress tracking for uploads
- Document management (list, delete)

Tech Stack: FastAPI, Supabase (pgvector), OpenAI
"""

import os
import json
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.db.supabase_client import supabase
from app.rag.ingestion import extract_text, chunk_text, generate_embeddings_batch
from app.rag.retriever import retrieve_similar_chunks
from app.rag.llm_chain import generate_answer_stream

# Initialize FastAPI application
app = FastAPI()

# In-memory dictionary to track upload progress
# Key: upload_id (UUID), Value: progress data dict
upload_progress = {}

# Configure CORS — in production, replace "*" with your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================================
# DOCUMENT MANAGEMENT
# ============================================================================

@app.get("/documents")
async def get_documents():
    """
    Retrieve all documents from the database.
    Supabase call is async-safe here because FastAPI runs sync route handlers
    in a thread pool automatically. Using async def for consistency.
    """
    import asyncio
    response = await asyncio.to_thread(
        lambda: supabase.table("documents").select("*").execute()
    )
    return response.data


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks (cascade delete via FK).
    """
    import asyncio
    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("documents").delete().eq("id", document_id).execute()
        )
        if response.data:
            return {"status": "success", "message": "Document deleted successfully"}
        return {"status": "error", "message": "Document not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# UPLOAD & PROCESSING
# ============================================================================

@app.get("/upload/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """
    Get real-time progress of a document upload/processing.
    Frontend polls this endpoint every 500ms to display the progress bar.
    """
    if upload_id not in upload_progress:
        return {"status": "not_found"}
    return upload_progress[upload_id]


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Handle file upload and queue background processing.

    Returns immediately with an upload_id so the frontend can start
    polling /upload/status/{upload_id} for progress.
    """
    file_path = f"../temp_{file.filename}"
    upload_id = str(uuid.uuid4())

    upload_progress[upload_id] = {
        "status": "uploading",
        "filename": file.filename,
        "progress": 0,
        "message": "Uploading file...",
        "total_chunks": 0,
        "processed_chunks": 0,
    }

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        upload_progress[upload_id]["status"] = "processing"
        upload_progress[upload_id]["progress"] = 10
        upload_progress[upload_id]["message"] = "File uploaded, starting processing..."

        # Background task runs in a thread pool — sync code is fine here
        background_tasks.add_task(
            process_file_background, file.filename, file_path, upload_id
        )

        return {
            "status": "processing",
            "upload_id": upload_id,
            "message": "File is being processed in the background",
        }

    except Exception as e:
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["message"] = str(e)
        return {"status": "error", "detail": str(e)}


# ============================================================================
# BACKGROUND PROCESSING (sync — runs in FastAPI's thread pool)
# ============================================================================

def process_file_background(filename: str, file_path: str, upload_id: str):
    """
    Process uploaded file in the background to avoid request timeouts.

    Steps:
    1. Extract text            (20%)
    2. Chunk text              (30%)
    3. Generate embeddings     (40–70%)
    4. Save to database        (70–100%)
    """
    try:
        upload_progress[upload_id]["progress"] = 20
        upload_progress[upload_id]["message"] = "Extracting text from document..."
        text = extract_text(file_path)

        upload_progress[upload_id]["progress"] = 30
        upload_progress[upload_id]["message"] = "Splitting into chunks..."
        chunks = chunk_text(text)
        upload_progress[upload_id]["total_chunks"] = len(chunks)

        upload_progress[upload_id]["progress"] = 40
        upload_progress[upload_id]["message"] = (
            f"Generating embeddings for {len(chunks)} chunks..."
        )

        save_chunks_with_progress(filename, chunks, upload_id)

        upload_progress[upload_id]["status"] = "complete"
        upload_progress[upload_id]["progress"] = 100
        upload_progress[upload_id]["message"] = (
            f"✅ Successfully processed {len(chunks)} chunks!"
        )
        print(f"✅ Successfully processed {filename}: {len(chunks)} chunks")

    except Exception as e:
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["progress"] = 0
        upload_progress[upload_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Error processing {filename}: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def save_chunks_with_progress(document_name: str, chunks: list, upload_id: str):
    """
    Save document chunks to Supabase with real-time progress updates.

    Phase 1 (40–70%): Batch-generate embeddings via OpenAI
    Phase 2 (70–100%): Batch-insert chunk rows into Supabase
    """
    doc_resp = supabase.table("documents").insert({"name": document_name}).execute()
    document_id = doc_resp.data[0]["id"]

    EMBEDDING_BATCH_SIZE = 100
    SUPABASE_BATCH_SIZE = 50

    all_rows = []
    total_chunks = len(chunks)

    # Phase 1: Embeddings
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = generate_embeddings_batch(batch)

        for chunk, embedding in zip(batch, embeddings):
            all_rows.append(
                {
                    "document_id": document_id,
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {"source": document_name},
                }
            )

        progress = 40 + int((len(all_rows) / total_chunks) * 30)
        upload_progress[upload_id]["progress"] = progress
        upload_progress[upload_id]["message"] = (
            f"Generated {len(all_rows)}/{total_chunks} embeddings..."
        )

    # Phase 2: Database inserts
    upload_progress[upload_id]["message"] = "Saving to database..."

    for i in range(0, len(all_rows), SUPABASE_BATCH_SIZE):
        batch_rows = all_rows[i : i + SUPABASE_BATCH_SIZE]

        for attempt in range(3):
            try:
                supabase.table("chunks").insert(batch_rows).execute()
                chunks_saved = min(i + SUPABASE_BATCH_SIZE, len(all_rows))
                progress = 70 + int((chunks_saved / len(all_rows)) * 30)
                upload_progress[upload_id]["progress"] = progress
                upload_progress[upload_id]["processed_chunks"] = chunks_saved
                upload_progress[upload_id]["message"] = (
                    f"Saved {chunks_saved}/{len(all_rows)} chunks to database..."
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                import time
                time.sleep(1)


# ============================================================================
# Q&A — STREAMING ENDPOINT
# ============================================================================

@app.get("/ask")
async def ask_question(question: str):
    """
    Answer a question using RAG, streaming the response via Server-Sent Events.

    Flow:
    1. Asynchronously retrieve the top-K relevant chunks (vector search)
    2. Immediately send a 'sources' event so the frontend can show citations
    3. Stream GPT-4o tokens as 'token' events — frontend renders them live
    4. Send a 'done' event so the frontend knows the stream is finished

    Each SSE event is a JSON object:
        {"type": "sources", "content": ["file.pdf", ...]}
        {"type": "token",   "content": "Hello"}
        {"type": "done"}
        {"type": "error",   "content": "...message..."}

    The 'Cache-Control' and 'X-Accel-Buffering' headers are required to
    prevent proxies (nginx, Cloudflare) from buffering the SSE stream.
    """
    relevant_chunks = await retrieve_similar_chunks(question, top_k=3)

    if not relevant_chunks:
        async def no_docs():
            event = {"type": "error", "content": "No relevant documents found. Please upload some files first."}
            yield f"data: {json.dumps(event)}\n\n"
        return StreamingResponse(
            no_docs(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Deduplicated source list — sent immediately before streaming starts
    sources = list(
        {c.get("metadata", {}).get("source", "Unknown") for c in relevant_chunks}
    )

    async def event_stream():
        # 1. Send sources right away so the UI can show them while text streams
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

        # 2. Stream answer tokens
        async for token in generate_answer_stream(question, relevant_chunks):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # 3. Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )