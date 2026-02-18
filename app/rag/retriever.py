from openai import OpenAI
import os
from app.db.supabase_client import supabase
from app.rag.ingestion import generate_embedding

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_similar_chunks(question: str, top_k=5):
    question_emb = generate_embedding(question)
    
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_emb,
        "match_threshold": 0.2, 
        "match_count": top_k
    }).execute()
    
    return result.data