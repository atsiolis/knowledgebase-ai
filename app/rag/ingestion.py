import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from app.db.supabase_client import supabase

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: # Check if page isn't an empty image
                    text += extracted + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text: str, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def generate_embedding(text: str):
    """Generate embedding for a single text (used by retriever)"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts  # Send all texts at once
    )
    # Return embeddings in the same order as input
    return [item.embedding for item in response.data]

def save_chunks(document_name: str, chunks: list):
    # 1. Save document entry
    doc_resp = supabase.table("documents").insert({"name": document_name}).execute()
    document_id = doc_resp.data[0]["id"]
    
    # 2. Process in smaller batches to avoid timeouts
    EMBEDDING_BATCH_SIZE = 100  # OpenAI batch limit
    SUPABASE_BATCH_SIZE = 50    # REDUCED: Embeddings are large, so insert fewer rows at once
    
    all_rows = []
    
    # Generate embeddings in batches
    print(f"Generating embeddings for {len(chunks)} chunks...")
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
        print(f"  Generated {len(all_rows)}/{len(chunks)} embeddings")
    
    # Insert to Supabase in small batches with retry logic
    print(f"Inserting {len(all_rows)} rows to Supabase...")
    for i in range(0, len(all_rows), SUPABASE_BATCH_SIZE):
        batch_rows = all_rows[i:i + SUPABASE_BATCH_SIZE]
        
        # Retry logic in case of transient failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                supabase.table("chunks").insert(batch_rows).execute()
                print(f"  Inserted batch {i//SUPABASE_BATCH_SIZE + 1}/{(len(all_rows)-1)//SUPABASE_BATCH_SIZE + 1}")
                break  # Success, move to next batch
            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # Give up after max retries
                print(f"  Retry {attempt + 1}/{max_retries} for batch {i//SUPABASE_BATCH_SIZE + 1}")
                import time
                time.sleep(1)  # Wait a bit before retrying