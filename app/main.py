import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.db.supabase_client import supabase
from app.rag.ingestion import extract_text, chunk_text, generate_embeddings_batch
from app.rag.retriever import retrieve_similar_chunks
from app.rag.llm_chain import generate_answer
from fastapi.middleware.cors import CORSMiddleware
import uuid

app = FastAPI()

# Progress tracking dictionary
upload_progress = {}

# Add this right after 'app = FastAPI()'
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, "*" allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents")
def get_documents():
    response = supabase.table("documents").select("*").execute()
    return response.data 

@app.get("/upload/status/{upload_id}")
def get_upload_status(upload_id: str):
    """Get real-time progress of an upload"""
    if upload_id not in upload_progress:
        return {"status": "not_found"}
    return upload_progress[upload_id]

@app.get("/ask")
async def ask_question(question: str):
    # Step 1: Find the relevant text in Supabase
    relevant_chunks = retrieve_similar_chunks(question, top_k=3)
    
    if not relevant_chunks:
        return {"answer": "No relevant documents found. Please upload some files first."}

    # Step 2: Generate the answer using OpenAI
    answer = generate_answer(question, relevant_chunks)
    
    return {
        "answer": answer,
        "sources": [c.get('metadata', {}).get('source', 'Unknown') for c in relevant_chunks]
    }

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = f"../temp_{file.filename}"
    upload_id = str(uuid.uuid4())
    
    # Initialize progress tracking
    upload_progress[upload_id] = {
        "status": "uploading",
        "filename": file.filename,
        "progress": 0,
        "message": "Uploading file...",
        "total_chunks": 0,
        "processed_chunks": 0
    }
    
    try:
        # Save uploaded file temporarily
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        upload_progress[upload_id]["status"] = "processing"
        upload_progress[upload_id]["progress"] = 10
        upload_progress[upload_id]["message"] = "File uploaded, starting processing..."
        
        # Process in background - return immediately!
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

def process_file_background(filename: str, file_path: str, upload_id: str):
    """Process file in background with progress updates"""
    try:
        # Extract text
        upload_progress[upload_id]["progress"] = 20
        upload_progress[upload_id]["message"] = "Extracting text from document..."
        text = extract_text(file_path)
        
        # Chunk text
        upload_progress[upload_id]["progress"] = 30
        upload_progress[upload_id]["message"] = "Splitting into chunks..."
        chunks = chunk_text(text)
        upload_progress[upload_id]["total_chunks"] = len(chunks)
        
        # Save chunks with progress callback
        upload_progress[upload_id]["progress"] = 40
        upload_progress[upload_id]["message"] = f"Generating embeddings for {len(chunks)} chunks..."
        
        save_chunks_with_progress(filename, chunks, upload_id)
        
        # Complete
        upload_progress[upload_id]["status"] = "complete"
        upload_progress[upload_id]["progress"] = 100
        upload_progress[upload_id]["message"] = f"✅ Successfully processed {len(chunks)} chunks!"
        
        print(f"✅ Successfully processed {filename}: {len(chunks)} chunks")
    except Exception as e:
        upload_progress[upload_id]["status"] = "error"
        upload_progress[upload_id]["progress"] = 0
        upload_progress[upload_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Error processing {filename}: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

def save_chunks_with_progress(document_name: str, chunks: list, upload_id: str):
    """Modified save_chunks that reports progress"""
    
    # Save document entry
    doc_resp = supabase.table("documents").insert({"name": document_name}).execute()
    document_id = doc_resp.data[0]["id"]
    
    EMBEDDING_BATCH_SIZE = 100
    SUPABASE_BATCH_SIZE = 50
    
    all_rows = []
    total_chunks = len(chunks)
    
    # Generate embeddings in batches with progress updates
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        embeddings = generate_embeddings_batch(batch)
        
        for chunk, embedding in zip(batch, embeddings):
            all_rows.append({
                "document_id": document_id,
                "content": chunk,
                "embedding": embedding,
                "metadata": {"source": document_name}
            })
        
        # Update progress (40-70% for embeddings)
        progress = 40 + int((len(all_rows) / total_chunks) * 30)
        upload_progress[upload_id]["progress"] = progress
        upload_progress[upload_id]["message"] = f"Generated {len(all_rows)}/{total_chunks} embeddings..."
    
    # Insert to Supabase in batches with progress updates
    upload_progress[upload_id]["message"] = "Saving to database..."
    for i in range(0, len(all_rows), SUPABASE_BATCH_SIZE):
        batch_rows = all_rows[i:i + SUPABASE_BATCH_SIZE]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                supabase.table("chunks").insert(batch_rows).execute()
                
                # Update progress (70-100% for database inserts)
                chunks_saved = min(i + SUPABASE_BATCH_SIZE, len(all_rows))
                progress = 70 + int((chunks_saved / len(all_rows)) * 30)
                upload_progress[upload_id]["progress"] = progress
                upload_progress[upload_id]["processed_chunks"] = chunks_saved
                upload_progress[upload_id]["message"] = f"Saved {chunks_saved}/{len(all_rows)} chunks to database..."
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(1)